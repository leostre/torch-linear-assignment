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
    'bf16': torch.bfloat16,
    'f32': torch.float32
}

REAL_DATA = torch.load('/root/torch-linear-assignment/data/train-start.pth')
ANS = torch.load('data/start-ans.pth')

def real_data(bs, type):
    if is_cuda:
        costs = REAL_DATA[:bs].to('cuda').to(_TYPES[type])
    assert costs.dtype == _TYPES[type]
    acc = 1

    try:
        torch.cuda.synchronize()
        t = time()
        result = batch_linear_assignment(costs)
        t = time() - t
        result_ref= ANS[:bs]
        acc = (result == result_ref) / result.numel()       
    except:
        return -1e-5, acc
    finally:
        del costs 
        # torch.cuda.empty_cache()
    return t, acc

print('STARTED BS EVALUATION'.center(80, '+'))
for TYPE in _TYPES:
    results = []
    REPS = 10
    bss = [4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16364]

    for bs in tqdm(bss):
        times = [] 
        accs = []
        for i in range(REPS):
            t, acc = real_data(bs, TYPE)
            times.append(t)
            accs.append(acc)
        times = np.array(times)
        accs = np.array(accs)
        results.append(
            (bs, times.mean(), times.std(), accs.mean(), accs.std())
        )

    columns = ['bs', 'time_mean', 'time_std', 'accs_mean', 'accs_std']
    resdf = pd.DataFrame(data=results, columns=columns)

    resdf.to_csv(f'experiments/results/bs_acc_{EXP_NAME}_{TYPE}.csv')



