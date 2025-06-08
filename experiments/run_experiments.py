from utils import get_current_git_branch 
import os

from time import time
from itertools import product

from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from torch_linear_assignment import batch_linear_assignment

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EXP_NAME = get_current_git_branch()

is_cuda = torch.cuda.is_available()
assert is_cuda, 'Requires CUDA!'

TYPES = {
    'f16': torch.float16, 
    'f32': torch.float32,
}

def nr_nc_ratio(nr, nc, factor=1, type=torch.float32):
    bs = 1
    torch.cuda.synchronize()
    costs = torch.arange(bs * nr * nc)[torch.randperm(bs * nr * nc)].reshape(bs, nr, nc) * factor
    if is_cuda:
        costs = costs.to('cuda').to(type)
    assert costs.dtype is type
    t = time()
    result_own = batch_linear_assignment(costs)
    t = time() - t
    
    
    result_ref = linear_sum_assignment(costs.to(torch.float32)[0].cpu().numpy())
    # except ValueError:
    #     print(f'{nr} x {nc} is infeasible')
    #     return np.nan, np.nan

    del costs 
    torch.cuda.empty_cache()

    return (result_own.cpu().numpy()[0, result_ref[0]] == result_ref[1]).sum() / len(result_ref[1]), t

print('Started SCALE'.center(80, '+'))

for tp in TYPES:
    results = []
    REPS = 20
    factors = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 
            1e-7, 1e-8, 
            #    1e-9, 1e-10,
            #     1e4, 1e8
    ]
    nrs = [4, 16, 64, 128]
    ncs = [4, 16, 64, 128]

    for nr, nc, factor in tqdm(product(nrs, ncs, factors)):
        # if nr > nc:
        #     continue
        times = [] 
        right_rate = []
        for i in range(REPS):
            try:
                rate, t = nr_nc_ratio(nr, nc, factor, type=TYPES[tp])
            except:
                rate = t = -1

            times.append(t)
            right_rate.append(rate)
        times = np.array(times)
        right_rate = np.array(right_rate)
        results.append(
            (nr, nc, factor, right_rate.mean(), right_rate.std(), times.mean(), times.std())
        )
        

    columns = ['nr', 'nc', 'factor', 'acc_mean', 'acc_std', 'time_mean', 'time_std']
    resdf = pd.DataFrame(data=results, columns=columns)

    resdf.to_csv(f'experiments/results/scale_{EXP_NAME}_{tp}.csv')



print('Started SHIFT'.center(80, '+'))
def nr_nc_ratio_shift(nr, nc, shift=0, type=torch.float32):
    bs = 1
    torch.cuda.synchronize()
    costs = torch.arange(bs * nr * nc)[torch.randperm(bs * nr * nc)].reshape(bs, nr, nc).float()
    costs.div_(bs * nr * nc - 1)
    costs.add_(shift)
    if is_cuda:
        costs = costs.to('cuda').to(type)
    t = time()
    result_own = batch_linear_assignment(costs)
    t = time() - t
    result_ref = linear_sum_assignment(costs[0].cpu().numpy())
    del costs 
    torch.cuda.empty_cache()
    return (result_own.cpu().numpy()[0, result_ref[0]] == result_ref[1]).sum() / len(result_ref[1]), t


for tp in TYPES:
    results_shift = []
    REPS = 20
    shifts = [1, 1e2, 1e4,
            #    1e6, 
            #   1e8, 1e10
            ]
    nrs = [4, 16, 64, 128]
    ncs = [4, 16, 64, 128]

    for nr, nc, shift in tqdm(product(nrs, ncs, shifts)):
        # if nr > nc:
        #     continue
        times = [] 
        right_rate = []
        for i in range(REPS):
            try:
                rate, t = nr_nc_ratio_shift(nr, nc, shift, type=TYPES[tp])
            except:
                rate = t = -1e-5
            times.append(t)
            right_rate.append(rate)
        times = np.array(times)
        right_rate = np.array(right_rate)
        results_shift.append(
            (nr, nc, shift, right_rate.mean(), right_rate.std(), times.mean(), times.std())
        )
        
    columns = ['nr', 'nc', 'shift', 'acc_mean', 'acc_std', 'time_mean', 'time_std']
    resdf_shift = pd.DataFrame(data=results_shift, columns=columns)

    resdf_shift.to_csv(f'experiments/results/shift_{EXP_NAME}_{tp}.csv')
