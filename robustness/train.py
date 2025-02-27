# from typing_extensions import Required
from numpy import zeros
import torch as ch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, Adam, Adamax
from torch.autograd import Variable
from torchvision.utils import make_grid
from cox.utils import Parameters
import numpy as np
from .tools import helpers
from .tools.helpers import AverageMeter, calc_fadein_eps, \
        save_checkpoint, ckpt_at_epoch, has_attr
from .tools import constants as consts
import dill
import time
import scipy
import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

def check_required_args(args, eval_only=False):
    """
    Check that the required training arguments are present.

    Args:
        args (argparse object): the arguments to check
        eval_only (bool) : whether to check only the arguments for evaluation
    """
    required_args_eval = ["adv_eval"]
    required_args_train = ["epochs", "out_dir", "adv_train",
        "log_iters", "lr", "momentum", "weight_decay"]
    adv_required_args = [] # ["attack_steps", "eps", "constraint", "use_best",
                        #"eps_fadein_epochs", "attack_lr", "random_restarts"]

    # Generic function for checking all arguments in a list
    def check_args(args_list):
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    # Different required args based on training or eval:
    if not eval_only: check_args(required_args_train)
    else: check_args(required_args_eval)
    # More required args if we are robustly training or evaling
    if bool(args.adv_train) or bool(args.adv_eval):
        check_args(adv_required_args)
    # More required args if the user provides a custom training loss
    has_custom_train = has_attr(args, 'custom_train_loss')
    has_custom_adv = has_attr(args, 'custom_adv_loss')
    if has_custom_train and not has_custom_adv:
        raise ValueError("Cannot use custom train loss \
            without a custom adversarial loss (see docs)")

def make_optimizer_and_schedule(args, model, checkpoint):
    """
    *Internal Function* (called directly from train_model)

    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.

    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`

    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    # Make optimizer
    optimizer = SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # Make schedule
    schedule = None
    
    if args.custom_schedule:
        cs = args.custom_schedule
        periods = eval(cs) if type(cs) is str else cs
        def lr_func(ep):
            for (milestone, lr) in reversed(periods):
                if ep >= milestone: return lr/args.lr
            return args.lr
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)

    elif args.base_lr and args.max_lr:
        schedule = lr_scheduler.CyclicLR(optimizer, args.base_lr, args.max_lr)

    elif args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()

    return optimizer, schedule

def eval_model(args, loader, model, store):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments. The required arguments are `adv_eval`,
            and `attack_lr`, `constraint`, `eps`, `attack_steps`, `random_restarts`
            if `adv_eval` is `1` or `True`.
        loader (iterable) : a dataloader serving `(input, label)` batches from the
            validation set
        model (AttackerModel) : model to evaluate
        store (cox.Store) : store for saving results in (via tensorboardX)
    """
    args.eps_fadein_epochs = 0 # By default
    check_required_args(args, eval_only=True)
    writer = store.tensorboard if store else None
    model = ch.nn.DataParallel(model).cuda()
    store.add_table('results', {'natacc':float, 'advacc':float})
    natacc, _ = _model_loop(args, 'val', loader, model, None, 0, False, writer)
    if args.adv_eval:
        advacc, _ = _model_loop(args, 'val', loader, model, None, 0, True, writer)

    store['results'].append_row({
        'natacc':natacc,
        'advacc':advacc
    })

def train_model(args, model, loaders, *, checkpoint=None, store=None):
    """
    Main function for training a model. 

    Args:
        args (object) : A python object for arguments, having attributes:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_schedule (str)
                If given, use a custom LR schedule (format: [(epoch, LR),...])
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            eps_fadein_epochs (int, *required if adv_train or adv_eval*)
                If greater than 0, fade in epsilon along this many epochs
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            regularizer (function, optional) 
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)` 
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        store (cox.Store) : a cox store for logging training progress
    """
    # Logging setup
    writer = store.tensorboard if store else None
    if store is not None: 
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
        store.add_table(consts.CKPTS_TABLE, consts.CKPTS_SCHEMA)
    
    # Reformat and read arguments
    check_required_args(args) # Argument sanity check
    args.eps = eval(str(args.eps)) if args.eps else None
    args.attack_lr = eval(str(args.attack_lr)) if args.attack_lr else None

    # Initial setup
    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint)
   
    best_prec1, start_epoch = (0, 0)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint[f"{'adv' if args.adv_train else 'nat'}_prec1"]

    # Put the model into parallel mode
    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model).cuda()

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train_prec1, train_loss = _model_loop(args, 'train', train_loader, 
                model, opt, epoch, args.adv_train, writer)
        last_epoch = (epoch == (args.epochs - 1))

        # evaluate on validation set
        sd_info = {
            'model':model.state_dict(),
            'optimizer':opt.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1
        }

        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else \
                                          store.path, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            prec1, nat_loss = _model_loop(args, 'val', val_loader, model, 
                        None, epoch, False, writer)
            # with ch.no_grad():
            #     prec1, nat_loss = _model_loop(args, 'val', val_loader, model, 
            #             None, epoch, False, writer)

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and _model_loop(args, 'val', val_loader,
                    model, None, epoch, True, writer)
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            is_best = our_prec1 > best_prec1
            best_prec1 = max(our_prec1, best_prec1)

            # log every checkpoint
            log_info = {
                'epoch':epoch + 1,
                'nat_prec1':prec1,
                'adv_prec1':adv_prec1,
                'nat_loss':nat_loss,
                'adv_loss':adv_loss,
                'train_prec1':train_prec1,
                'train_loss':train_loss
            } 

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch))
            # If  store exists and this is the last epoch, save a checkpoint
            if last_epoch and store: store[consts.CKPTS_TABLE].append_row(sd_info)

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)

        if schedule: schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(model, log_info)

    return model

def _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    if is_train:
        scaler = ch.cuda.amp.GradScaler()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = calc_fadein_eps(epoch, args.eps_fadein_epochs, args.eps) \
                if is_train else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
            else ch.nn.CrossEntropyLoss().cuda()

    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    loss_kwargs = {}
    
    if adv:
            attack_kwargs = {
            'rot': args.rot,
            'trans': args.trans, 
            'scale': args.scale,
            'hue': args.hue,
            'satu': args.satu,
            'bright': args.bright,
            'cont': args.cont,  
            'gau_size': args.gau_size,
            'gau_sigma': args.gau_sigma,
            'transform_type': args.transform_type,
            'attack_type': args.attack_type,
            'do_tqdm': False,
            'tries':args.tries,
            'use_best': args.use_best
        } 
        
    loss_kwargs = {
            'rot': args.rot,
            'trans': args.trans, 
            'scale': args.scale,
            'hue': args.hue,
            'satu': args.satu,
            'bright': args.bright,
            'cont': args.cont,  
            'gau_size': args.gau_size,
            'gau_sigma': args.gau_sigma,
            'transform_type': args.transform_type,
            'attack_type': args.attack_type,
            'do_tqdm': False,
            'tries':1,
            'use_best': 0
        }

    iterator = tqdm(enumerate(loader), total=len(loader))


    for i, (inp, target) in iterator:
       # measure data loading time
        target = target.cuda(non_blocking=True)

        # with ch.autograd.detect_anomaly(True):
        
        output, final_inp = model(inp, target=target, make_adv=adv,
                                  **attack_kwargs)
                                  
        loss = train_criterion(model, inp, target, loss_kwargs) if has_custom_train_loss else train_criterion(output, target)

        # print("training loss:", loss)

        if len(loss.shape) > 0: loss = loss.mean()
        model_logits = output[0] if (type(output) is tuple) else output

        # measure accuracy and record loss
        maxk = min(5, model_logits.shape[-1])
        prec1, prec5 = helpers.accuracy(model_logits, target, topk=(1, maxk))

        losses.update(loss.item(), inp.size(0))
        top1.update(prec1[0], inp.size(0))
        top5.update(prec5[0], inp.size(0))

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term =  args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            loss.backward()
            ch.nn.utils.clip_grad_norm_(model.parameters(), 1e5)
            opt.step()

        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            writer.add_image('Nat input', nat_grid, epoch)
            writer.add_image('Adv input', adv_grid, epoch)

        # ITERATOR
        desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                '{1}1 {top1.avg:.3f} | {1}5 {top5.avg:.3f} | '
                'Reg term: {reg} ||'.format( epoch, prec, loop_msg, 
                loss=losses, top1=top1, top5=top5, reg=reg_term))

        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                              epoch)
    ch.cuda.empty_cache()
    return top1.avg, losses.avg

