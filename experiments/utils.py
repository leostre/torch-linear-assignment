import torch
from tqdm import tqdm
import time
import os 

import numpy as np

from torch_linear_assignment import batch_linear_assignment

import os
import random

def set_all_seeds(seed=42, multi_gpu=False):
    """
    Set the seed for all random number generators to ensure reproducibility.
    
    Parameters:
        seed (int): The seed value to use (default: 42)
    """
    # Python built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # CuDNN (can impact performance but ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"All seeds set to {seed}")


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


def mae(pred, targ):
    return torch.mean(torch.abs(pred - targ))

def rmse(pred, targ):
    return torch.sqrt(torch.mean(torch.square(pred- targ)))

def smape(pred, targ):
    return 200 * torch.mean(torch.abs(pred - targ) / (torch.abs(pred) + torch.abs(targ)))


def sum_error(X: torch.Tensor, ans_indices: torch.Tensor, res_indices: torch.Tensor):
    worker_idx = torch.arange(X.size(1))
    s = torch.zeros(X.size(0))
    for i, (cost, res_ind) in enumerate(zip(X, res_indices)):
        s[i] = cost[worker_idx, res_ind].sum()

    sans = torch.zeros(X.size(0))
    for i, (cost, res_ind) in enumerate(zip(X, ans_indices)):
        sans[i] = cost[worker_idx, res_ind].sum()
    return (s - sans).numpy(), mae(s, sans).item(), rmse(s, sans).item(), smape(s, sans).item()



import subprocess

def get_current_git_branch():
    """
    Returns the name of the current Git branch.
    
    Returns:
        str: Name of the current Git branch, or None if not in a Git repo or if detached HEAD.
    """
    try:
        # Get the current branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        branch = result.stdout.strip()
        
        # Handle detached HEAD state (returns "HEAD")
        if branch == "HEAD":
            return None
            
        return branch
    except subprocess.CalledProcessError:
        # Not a Git repository or other error
        return None
    except FileNotFoundError:
        # Git command not found
        return None


# Example usage
if __name__ == "__main__":
    branch = get_current_git_branch()
    if branch:
        print(f"Current branch: {branch}")
    else:
        print("Not in a Git repository or in detached HEAD state")


