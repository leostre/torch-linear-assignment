import time

from itertools import product

import torch

from utils import *

OUTPUT_FILE = f'/root/torch-linear-assignment/experiments/results_{last_commit_hash()}.csv'
REPS = 5
BSS = product(
    [4, 16, 64, 256, 1024],
    [16, 64, 256, 1024],
    [16, 64, 256, 1024]
)

assert torch.cuda.is_available(), 'Expected CUDA usage'
torch.set_default_device('cuda')

if __name__ == '__main__':
    run(BSS, OUTPUT_FILE, reps=REPS)
