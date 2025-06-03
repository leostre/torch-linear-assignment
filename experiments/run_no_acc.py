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

is_cuda = torch.cuda.is_available()
assert is_cuda, 'Requires CUDA!'


os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

set_all_seeds()

EXP_NAME = get_current_git_branch()

# TYPE = 'f16'
_TYPES = {
    'f16': torch.float16,
    'f32': torch.float32
}

real_costs = torch.load('/root/torch-linear-assignment/data/train-end.pth')

def nr_nc(bs, type):
    torch.cuda.synchronize()
    if is_cuda:
        costs = real_costs[:bs].to('cuda').to(_TYPES[type])
    assert costs.dtype == _TYPES[type]
    t = time()
    batch_linear_assignment(costs)
    t = time() - t
    
    del costs 
    torch.cuda.empty_cache()

    return t

print('STARTED BS EVALUATION'.center(80, '+'))
for TYPE in _TYPES:
    results = []
    REPS = 10
    bss = [4, 16, 32, 64, 128, 256, 512, 1024]

    for bs in tqdm(bss):
        times = [] 
        for i in range(REPS):
            t = nr_nc(bs, TYPE)
            times.append(t)
        times = np.array(times)
        results.append(
            (bs, times.mean(), times.std())
        )

    columns = ['bs', 'time_mean', 'time_std']
    resdf = pd.DataFrame(data=results, columns=columns)

    resdf.to_csv(f'experiments/results/bs_{EXP_NAME}_{TYPE}.csv')
