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

_TYPES = {
    'f16': torch.float16,
    'bf16': torch.bfloat16,
    'f32': torch.float32
}
REPS = 20
BSS = [
    # 4, 16, 32, 64, 128,
    256, 
    # 512, 1024, 2048, 4096, 8192, 16364
    ]

EXP2 = [4, 16, 64, 128, 512, 1024,]
FACTORS = sorted([
    1,
    *EXP2,
    *[exp - 1 for exp in EXP2],
    *[int(0.6 * exp) for exp in EXP2],
    ]
)

real_costs = torch.load('/root/torch-linear-assignment/data/train-start.pth').to('cuda')
ans = torch.load('data/start-ans.pth').to('cuda')

def sca_bs(costs, bs, factor, type):
    costs = costs[:bs].to(_TYPES[type])
    assert costs.dtype == _TYPES[type]
    acc = 1e-5
    try:
        torch.cuda.synchronize()
        t = time()
        res = batch_linear_assignment(costs, factor)
        t = time() - t     
        acc = ((res == ans[:bs]).sum() / res.numel()).cpu().numpy()
    except Exception as x:
        print(x)
        return -1e-5, -1e-5
    finally:
        del costs 
        torch.cuda.empty_cache()
    return t, acc

def run(costs, TYPE, name=''):
        results = []
        REPS = 20

        for bs, factor in tqdm(product(BSS, FACTORS)):
            times = [] 
            accs = []
            for i in range(REPS):
                t, acc = sca_bs(costs, bs, factor, TYPE)
                times.append(t)
                accs.append(acc)
            times = np.array(times)
            accs = np.array(accs)
            results.append(
                (bs, factor, times.mean(), times.std(), accs.mean(), accs.std())
            )

        columns = ['bs', 'factor', 'time_mean', 'time_std', 'acc_mean', 'acc_std']
        resdf = pd.DataFrame(data=results, columns=columns)

        resdf.to_csv(f'experiments/results/scacc{name}_{EXP_NAME}_{TYPE}.csv')

print('STARTED ACC SCALE EVALUATION'.center(80, '+'))
for TYPE in _TYPES:
    run(real_costs, TYPE)

art_x = torch.load('/root/torch-linear-assignment/data/art_x.pth').to('cuda')
art_ans = torch.load('/root/torch-linear-assignment/data/art_ans.pth').to('cuda')


print('STARTED ACC SCALE EVALUATION ## ART'.center(80, '+'))
for TYPE in _TYPES:
    run(art_x, TYPE, name='_art')
    
