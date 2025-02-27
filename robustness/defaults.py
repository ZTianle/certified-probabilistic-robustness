from xmlrpc.client import boolean
from . import attacker, datasets
from .tools import helpers

BY_DATASET = 'varies by dataset'
REQ = 'REQUIRED'

TRAINING_DEFAULTS = {
    datasets.CIFAR: {
        "epochs": 110,
        "batch_size": 128,
        "weight_decay":5e-4,
        "step_lr": 50,
        # "base_lr": 1e-6,
        # "max_lr": 1e-5,
        # "custom_schedule": None,
        "base_lr": None,
        "max_lr": None,
        # "custom_schedule": [(0, 1e-5), (50, 1e-6), (100, 1e-7), (105, 1e-6)],
        # "custom_schedule": [(0, 1e-3), (40, 1e-3), (65, 1e-4), (90, 1e-5), (105, 1e-6)]
        # "base_lr": None,
        # "max_lr": None,
        "custom_schedule": [(0, 1e-1), (20, 1e-2), (40, 1e-3), (60, 1e-4), (80, 1e-5), (100, 1e-6)]
    },
        datasets.CIFAR100: {
        "epochs": 110,
        "batch_size": 128,
        "weight_decay":5e-4,
        "step_lr": 50,
        "base_lr": None,
        "max_lr": None,
        "custom_schedule": [(0, 1e-1), (20, 1e-2), (40, 1e-3), (60, 1e-4), (80, 1e-5), (100, 1e-6)]
    },
    datasets.CINIC: {
        "epochs": 150,
        "batch_size": 128,
        "weight_decay":5e-4,
        "step_lr": 50,
        "custom_schedule": None
    },
    datasets.ImageNet: {
        "epochs": 200,
        "batch_size":256,
        "weight_decay":1e-4,
        "step_lr": 50,
        "base_lr": None,
        "max_lr": None,
        "custom_schedule":[(0, .1), (30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]
    },
    datasets.RestrictedImageNet: {
        "epochs": 110,
        "batch_size": 256,
        "weight_decay": 1e-4,
        "step_lr": 50,
        "base_lr": None,
        "max_lr": None,
        "custom_schedule": [(0, .1), (30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]
    },
    datasets.A2B: {
        "epochs": 150,
        "batch_size": 64,
        "weight_decay": 5e-4,
        "step_lr": 50,
        "custom_schedule": None
    }
}

TRAINING_DEFAULTS[datasets.ImageNetNoCrop] = TRAINING_DEFAULTS[datasets.ImageNet]

"""
Default hyperparameters for training by dataset (tested for resnet50).
Parameters can be accessed as `TRAINING_DEFAULTS[dataset_class][param_name]`
"""

TRAINING_ARGS = [
    ['out-dir', str, 'where to save training logs and checkpoints', REQ],
    ['epochs', int, 'number of epochs to train for', BY_DATASET],
    ['lr', float, 'initial learning rate for training', 0.1],
    ['weight_decay', float, 'SGD weight decay parameter', BY_DATASET],
    ['momentum', float, 'SGD momentum parameter', 0.9],
    ['step-lr', int, 'number of steps between 10x LR drops', BY_DATASET],
    ['base-lr', float, 'Initial learning rate, i.e., the lower boundary in the CyclicLR', BY_DATASET],
    ['max-lr', float, ' Upper learning rate boundaries in the CyclicLR', BY_DATASET],
    ['custom-schedule', str, 'LR sched (format: [(epoch, LR),...])', BY_DATASET],
    ['adv-train', [0, 1], 'whether to train adversarially', REQ],
    ['adv-eval', [0, 1], 'whether to adversarially evaluate', None],
    ['log-iters', int, 'how frequently (in epochs) to log', 5],
    ['save-ckpt-iters', int, 'how frequently (epochs) to save \
    (-1 for none, only saves best and last)', -1],
    ['lam', float, 'balance of original loss and constraint loss', 0.1],
    ['t', float, 'confidence', 0.0],
    ['eps', float, 'violation rate', 0.00001],
]
"""
Arguments essential for the `train_model` function.

*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

PGD_ARGS = [
    ['tries', int, 'number of tries for attack', REQ],
    ['rot', int , '(recommanded 30) deg of rotations', REQ],
    ['trans', float , '(recommanded 10%) translations', REQ],
    ['scale', float , '(recommanded 0.3) scale', REQ],
    ['hue', float , '(recommanded pi/2) hue', REQ],
    ['satu', float , '(recommanded 0.3) saturation', REQ],
    ['bright', float , '(recommanded 0.3) brightness', REQ],
    ['cont', float , '(recommanded 0.3) contrast', REQ],
    ['gau-size', int , '(recommanded 11) brightness', REQ],
    ['gau-sigma', float , '(recommanded 1.5) contrast', REQ],
    ['use-best', [0, 1], 'if 1 (0) use best (final) PGD step as example', REQ],
    ['transform-type', ['spatial', 'color', 'blur', 'semantic'], 'spatial functional perturbation if spatial, color functional perturbation if spatial,  blurring otherwise', REQ],
    ['attack-type', ['grid', 'random'], 'grid if exhaustive, random attacker otherwise', REQ]
]
"""
Arguments essential for the :meth:`robustness.train.train_model` function if
adversarially training or evaluating.

*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

MODEL_LOADER_ARGS = [
    ['dataset', list(datasets.DATASETS.keys()), '', REQ],
    ['subset', int,'the number of training data points to use', None],
    ['data', str, 'path to the dataset', '/tmp/'],
    ['arch', str, 'architecture (see {cifar,imagenet}_models/', REQ],
    ['batch-size', int, 'batch size for data loading', BY_DATASET],
    ['workers', str, '# data loading workers', 16],
    ['resume', str, 'path to checkpoint to resume from', None],
    ['data-aug', [0, 1], 'whether to use data augmentation', 1],
    ['aug-method', ["RandAugment", "TrivialAugment", "AugMix"], 
                    'determine data augmentation approach, otherwise default', None],
]
"""
Arguments essential for constructing the model and dataloaders that will be fed
into :meth:`robustness.train.train_model` or :meth:`robustness.train.eval_model`

*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

CONFIG_ARGS = [
    ['config-path', str, 'config path for loading in parameters', None],
    ['eval-only', [0, 1], 'just run evaluation (no training)', 0],
    ['exp-name', str, 'where to save in (inside out_dir)', None]
]
"""
Arguments for main.py specifically

*Format*: `[NAME, TYPE/CHOICES, HELP STRING, DEFAULT (REQ=required,
BY_DATASET=looked up in TRAINING_DEFAULTS at runtime)]`
"""

def add_args_to_parser(arg_list, parser):
    """
    Adds arguments from one of the argument lists above to a passed-in
    arparse.ArgumentParser object. Formats helpstrings according to the
    defaults, but does NOT set the actual argparse defaults (*important*).

    Args:
        arg_list (list) : A list of the same format as the lists above, i.e.
            containing entries of the form [NAME, TYPE/CHOICES, HELP, DEFAULT]
        parser (argparse.ArgumentParser) : An ArgumentParser object to which the
            arguments will be added

    Returns:
        The original parser, now with the arguments added in.
    """
    for arg_name, arg_type, arg_help, arg_default in arg_list:
        has_choices = (type(arg_type) == list) 
        kwargs = {
            'type': type(arg_type[0]) if has_choices else arg_type,
            'help': f"{arg_help} (default: {arg_default})"
        }
        if has_choices: kwargs['choices'] = arg_type
        parser.add_argument(f'--{arg_name}', **kwargs)
    return parser

def check_and_fill_args(args, arg_list, ds_class):
    """
    Fills in defaults based on an arguments list (e.g., TRAINING_ARGS) and a
    dataset class (e.g., datasets.CIFAR).

    Args:
        args (object) : Any object subclass exposing :samp:`setattr` and
            :samp:`getattr` (e.g. cox.utils.Parameters)
        arg_list (list) : A list of the same format as the lists above, i.e.
            containing entries of the form [NAME, TYPE/CHOICES, HELP, DEFAULT]
        ds_class (type) : A dataset class name (i.e. a
            :class:`robustness.datasets.DataSet` subclass name)

    Returns:
        The :samp:`args` object with all the defaults filled in according to
            :samp:`arg_list` defaults.
    """
    for arg_name, _, _, arg_default in arg_list:
        name = arg_name.replace("-", "_")
        if helpers.has_attr(args, name): continue
        if arg_default == REQ: raise ValueError(f"{arg_name} required")
        elif arg_default == BY_DATASET:
            setattr(args, name, TRAINING_DEFAULTS[ds_class][name])
        elif arg_default is not None: 
            setattr(args, name, arg_default)
    return args



