from utils import get_current_git_branch, set_all_seeds

import os
from time import time
from itertools import product

from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from torch_linear_assignment import batch_linear_assignment

set_all_seeds()

is_cuda = torch.cuda.is_available()
assert is_cuda, 'Requires CUDA!'


os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EXP_NAME = get_current_git_branch()

# TYPE = 'f16'
_TYPES = {
    'f16': torch.float16,
    'f32': torch.float32
}

real_costs = torch.load('/root/torch-linear-assignment/data/train-start.pth')

def nr_nc(cost, bs, type):
    if is_cuda:
        costs = cost[:bs].to('cuda').to(_TYPES[type])
    assert costs.dtype == _TYPES[type]
    try:
        torch.cuda.synchronize()
        t = time()
        batch_linear_assignment(costs)
        t = time() - t
    except:
        return -1e-5
    finally:
        del costs 
        torch.cuda.empty_cache()
    return t

REPS = 20
BSS = [4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

def run(cost, tp):
    results = []
    for bs in tqdm(BSS):
        times = [] 
        for i in range(REPS):
            t = nr_nc(cost, bs, tp)
            times.append(t)
        times = np.array(times)
        results.append(
            (bs, times.mean(), times.std())
        )

    columns = ['bs', 'time_mean', 'time_std']
    resdf = pd.DataFrame(data=results, columns=columns)

    resdf.to_csv(f'experiments/results/bs_{EXP_NAME}_{tp}.csv')

if __name__ == '__main__':
    print('STARTED BS EVALUATION'.center(80, '+'))
    for TYPE in _TYPES:
        run(real_costs, TYPE)
