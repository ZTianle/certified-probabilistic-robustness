import math
import scipy
import torch as ch
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable
from robustness.attack_steps import transform
from robustness.datasets import CIFAR
from robustness.attacker import AttackerModel
from robustness.tools.vis_tools import show_image_row
from robustness.tools.label_maps import CLASS_DICT
from robustness.model_utils import make_and_restore_model
from robustness.preact_resnet import PreActResNet18
from statsmodels.stats.proportion import proportion_confint 

ds = CIFAR('./path/to/cifar', std = ch.tensor([0.2471, 0.2435, 0.2616]))
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm
# Returns the (epsilon, delta) values of the adaptive concentration inequality
# for the given parameters.
#
# n: int (number of samples)
# delta: float (parameter delta)
# return: epsilon (parameter epsilon)
def get_type(n, delta):
    n = float(n)
    b = -np.log(delta / 24.0) / 1.8
    epsilon = np.sqrt((0.6 * np.log(np.log(n)/np.log(1.1) + 1) + b) / n)
    return epsilon

# Returns the (epsilon, delta) values of the adaptive concentration inequality
# for the given parameters.
#
# n: int (number of samples)
# delta: float (parameter delta)
# return: epsilon (parameter epsilon)
def get_type_ho(n, delta):
    n = float(n)
    epsilon = np.sqrt((np.log(2)-np.log(delta)) / (2 * n))
    return epsilon

# Return required number of samples
#  tau: float (threshold of robustness)
# delta: float (parameter delta)  
def get_num_ho(tau, delta):
    n = -(np.log(delta / 2)) / (2 * (tau**2))
    return n

# Run type inference to get the type judgement for the robustness criterion.
# tau: float (threshold of robustness)
# n: int (number of samples)
# delta: float (parameter delta)
# E_Z: float (estimate of Z)
# Delta: float (threshold on inequalities)
def get_verification_type(tau, n, delta, E_Z, Delta=0):
    # Step 1: Get (epsilon, delta) values from the adaptive concentration inequality for the current number of samples
    epsilon = get_type(n, delta)

    # Step 2: Check if robustness holds
    if E_Z - epsilon >= 1-tau:
        return 1

    # Step 3: Check if robustness does not hold
    if E_Z + epsilon < 1-tau:
        return 0

    # Step 4: Check if robustness holds (ambiguously)
    if E_Z - epsilon >= 1-tau - Delta and epsilon <= Delta:
        return 1

    # Step 5: Check if robustness does not hold (ambiguously)
    if E_Z + epsilon < 1-tau + Delta and epsilon <= Delta:
        return 0

    # Step 6: Continue sampling
    return None

def get_verification_ho(tau, n, delta, E_Z, Delta=0):
    # Step 1: Get (epsilon, delta) values from the Hoeffding inequality for the current number of samples
    epsilon = get_type_ho(n, delta)

    # Step 2: Check if robustness holds
    if E_Z - epsilon >= 1-tau:
        return 1

    # Step 3: Check if robustness does not hold
    if E_Z + epsilon < 1-tau:
        return 0

    # Step 4: Check if robustness holds (ambiguously)
    if E_Z - epsilon >= 1-tau - Delta and epsilon <= Delta:
        return 1

    # Step 5: Check if robustness does not hold (ambiguously)
    if E_Z + epsilon < 1-tau + Delta and epsilon <= Delta:
        return 0

    # Step 6: Continue sampling
    return None

def get_verification_AC(tau, n, delta, Z_sum, Delta=0):
    ci_low, ci_upp = proportion_confint(Z_sum, n, method='agresti_coull', alpha=delta)
    # Step 2: Check if robustness holds
    
    if ci_low >= 1-tau:
        return 1

    # Step 3: Check if robustness does not hold
    if ci_upp < 1-tau:
        return 0

    # Step 4: Check if robustness holds (ambiguously)
    if ci_low >= 1-tau - Delta:
        return 1

    # Step 5: Check if robustness does not hold (ambiguously)
    if ci_upp < 1-tau + Delta:
        return 0

    # Step 6: Continue sampling
    return None


def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()

def margin_ind(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    # loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)
    return ind

def p_ind(x, y, d):
    ind = ch.lt(ch.norm(x-y, p=float('inf'), dim=1), d)
    return ind

def verify(X, label, model, type="adaptive", tau=0.01, delta=1e-10, sample_limit=10000, bs=200, kwargs={
    'rot': 15,
    'trans': 0.3, 
    'scale': 0.3,
    'hue': math.pi/4,
    'satu': 0.3,
    'bright': 0.3,
    'cont': 0.3,
    'tries':1,
    'use_best': True, 
    'transform_type': 'spatial',
    'attack_type': 'random',
    'do_tqdm': False
    }):
    # kwargs = {
    # # 'rot': 0,
    # 'trans': 0, 
    # 'scale': 0,
    # 'rot': 15,
    # # 'trans': 10, 
    # # 'scale': 0.3,
    # 'tries':1,
    # 'use_best': True, 
    # 'attack_type': 'random',
    # 'do_tqdm': False
    # }

    # Step 1: Initialization
    nE_p=0.0
    # all_label = label.repeat(bs)
    # Step 2: Calculate threshold d
    output_ori, _ = model(X)
    output_ori = F.softmax(output_ori, dim=1)
    output_sorted, ind_sorted = output_ori.sort(dim=1)
    d = (output_sorted[:, -1]-output_sorted[:, -2])/2  
    if type == "adaptive":
    # Step 3: Iteratively sample and check whether fairness holds
        Z_sum = 0
        all_label = label.repeat(bs)
        for i in range(int(sample_limit/bs)):
            with ch.no_grad(): 
                im = X.repeat(bs, 1, 1, 1)
                _, im_spat =model(im, all_label, make_adv=True, **kwargs)

                output, _ = model(im_spat)
                all_output = F.softmax(output, dim=1)
                # Z = ch.lt(ch.norm(output_ori-all_output, p=float('inf'), dim=1), d)
                Z = p_ind(output_ori, all_output, d)
                # Z = margin_ind(output, all_label)
                Z_sum += Z.sum()
                n = (i+1) * bs
                E_Z = Z_sum / n
                t = get_verification_type(tau, n, delta, E_Z)
                # Return if converged
                if not t is None:
                    return t, n
    # Step 4: Failed to verify after maximum number of samples
        return None, n
        
    elif type == "hoeffding":
        Z_sum = 0
        sample_limit = get_num_ho(tau, delta)
        
        if sample_limit < bs:
            bs = int(sample_limit)
        # print(sample_limit)
        all_label = label.repeat(bs)

        # for i in range(math.ceil(sample_limit/bs)):
        for i in range(round(sample_limit/bs)):
            with ch.no_grad(): 
                im = X.repeat(bs, 1, 1, 1)
                _, im_spat =model(im, all_label, make_adv=True, **kwargs)
                output, _ = model(im_spat)
                all_output = F.softmax(output, dim=1)
                Z = p_ind(output_ori, all_output, d)
                # Z = margin_ind(output, all_label)
                Z_sum += Z.sum()
        n = (math.ceil(sample_limit/bs)) * bs
        E_Z = Z_sum / n
        t = get_verification_ho(tau, n, delta, E_Z)
        # Return if converged
        if not t is None:
            return t, n
        else:
            return None, n
    
    elif type == "AC":
        Z_sum = 0

        all_label = label.repeat(bs)

        for i in range(math.ceil(sample_limit/bs)):
            with ch.no_grad(): 
                im = X.repeat(bs, 1, 1, 1)
                _, im_spat =model(im, all_label, make_adv=True, **kwargs)
                output, _ = model(im_spat)
                all_output = F.softmax(output, dim=1)
                Z = p_ind(output_ori, all_output, d)
                Z_sum += Z.sum()
        n = (math.ceil(sample_limit/bs)) * bs
        # E_Z = Z_sum / n
        # print(tau, n, delta, Z_sum.cpu())
        t = get_verification_AC(tau, n, delta, Z_sum.cpu())
        # Return if converged
        if not t is None:
            return t, n
        else:
            return None, n
    # elif type == "hoeffding":
    #     Z_sum = 0
    #     sample_limit = get_num_ho(tau, delta)
        
    #     if sample_limit < bs:
    #         bs = sample_limit

    #     # bs = int(bs)

    #     for i in range(math.ceil(sample_limit/bs)):
    #         with ch.no_grad(): 
    #             im = X.repeat(bs, 1, 1, 1)
    #             _, im_spat =model(im, all_label, make_adv=True, **kwargs)
    #             output, _ = model(im_spat)
    #             all_output = F.softmax(output, dim=1)
    #             Z = p_ind(output_ori, all_output, d)
    #             # Z = margin_ind(output, all_label)
    #             Z_sum += Z.sum()

        # rest = int(sample_limit) - int(sample_limit/bs) * bs
        # if rest > 0:
        #     with ch.no_grad(): 
        #         im = X.repeat(rest, 1, 1, 1)
        #         all_label = label.repeat(rest)
        #         _, im_spat =model(im, all_label, make_adv=True, **kwargs)
        #         output, _ = model(im_spat)
        #         all_output = F.softmax(output, dim=1)
        #         Z = p_ind(output_ori, all_output, d)
        #         # Z = margin_ind(output, all_label)
        #         Z_sum = Z_sum+Z.sum()

        # # n = (int(sample_limit/bs)) * bs + rest
        # n = (math.ceil(sample_limit/bs)) * bs 
        # E_Z = Z_sum / n
        # # print(tau, n, delta, E_Z)
        # t = get_verification_ho(tau, n, delta, E_Z)
        # # Return if converged
        # if not t is None:
        #     return t, n
        # else:
        #     return None, n

def cert(X, label, model, eps=1e-10, sample_limit=1000, bs=200, k=30):
        kwargs = {
        'spatial_constraint': 45,
        'tries':1,
        'use_best': True, 
        'attack_type': 'random',
        'do_tqdm': False
        }
        delta = 0.9
        all_label = label.repeat(bs)
        all_output = ch.ones(sample_limit, 10).cuda()
        output_ori, _ = model(X)
        output_ori = F.softmax(output_ori, dim=1)
        output_sorted, ind_sorted = output_ori.sort(dim=1)
        d = (output_sorted[:, -1]-output_sorted[:, -2])/2 
        output_ori = output_ori.repeat(sample_limit, 1)
        t = ch.linspace(1e-4, 1e4, 500).cuda()
        bounds = []
        for i in range(k):
            for i in range(int(sample_limit/bs)):
                with ch.no_grad():    
                    im = X.repeat(bs, 1, 1, 1)
                    _, im_spat =model(im, all_label, make_adv=True, **kwargs)
                    output, _ = model(im_spat)
                    all_output[bs*i:bs*(i+1), :] = F.softmax(output, dim=1)
            z = ch.norm(output_ori-all_output, p=float('inf'), dim=1)
            # outer = ch.outer(z,t)
            # e = ch.exp(d*t)
            # print(e)
            # E = ch.mean(ch.exp(outer), dim=0)
            # b = E / e
            outer = ch.outer(z,t)
            b = ch.mean(ch.exp(outer-d*t), dim=0)
            bounds.append(ch.min(b))
        bound = max(bounds) / delta

        if min(1.0, bound)<eps:
            return 1, sample_limit*k
        else:
            return 0, sample_limit*k
        # T = d

        # alpha = ch.tensor([1.], requires_grad = True)
        # def min_obj(alpha):
        #     alpha = ch.tensor(alpha).cuda()
        #     alpha.requires_grad_()
        #     g_c = ch.exp(alpha*(g - T)).mean()-eps
        #     print("alpha:", alpha, "gc:", g_c)
        #     min = g_c 
        #     min.backward(retain_graph=True)
        #     return min.data.cpu().numpy(), alpha.grad.data.cpu()
        # result = scipy.optimize.basinhopping(min_obj,
        #                             alpha.detach().numpy(), niter=10,
        #                             minimizer_kwargs={"method":"L-BFGS-B", "jac":True, "bounds":[(0, None)]}
        #                             ) # NB: we will compute the jacobian
        # print(result.x, g-T, result.fun)
        # return ch.tensor(max(0, result.fun))