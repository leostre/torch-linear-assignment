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
ACC = True

nc = nr = 8

def nr_nc(bs, type, acc=False):
    total = bs * nc * nr
    costs = torch.arange(total, dtype=torch.float)[torch.randperm(total)].reshape(bs, nr, nc).contiguous()
    if is_cuda:
        costs = costs.to('cuda').to(_TYPES[type])
    assert costs.dtype == _TYPES[type]
    try:
        torch.cuda.synchronize()
        t = time()
        result = batch_linear_assignment(costs)[0]
        t = time() - t
        acc = 1
        if acc:
            result_ref= batch_linear_assignment(costs.to('cpu'))[0]
        acc = (result == result_ref) / len(result)       

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
            t, acc = nr_nc(bs, TYPE)
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



