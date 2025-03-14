import torch
from tqdm import tqdm
import time
import os 

import numpy as np

from torch_linear_assignment import batch_linear_assignment

def last_commit_hash():
    os.system('git log -n 1 $(git rev-parse --abbrev-ref HEAD) > tmp.txt')
    with open('tmp.txt', 'rt') as file:
        hash_ = file.read().split('\n')[0].split()[-1]
    os.remove('tmp.txt')
    return hash_


def _run_one(bs):
    torch.cuda.synchronize()
    cost = torch.randn(*bs)
    start = time.time()
    batch_linear_assignment(cost)
    total = time.time() - start
    return total   


def run(bss, output_file, reps=10):
    with open(output_file, 'w+') as file:
        print('batch_size', 'workers', 'tasks', 'mean', 'std', sep=',', file=file)
    for bs in tqdm(bss, desc='Batch # '):
        results = [_run_one(bs) for _ in range(reps)]
        with open(output_file, 'at') as file:
            print(*bs, np.mean(results), np.std(results), sep=',', file=file)
