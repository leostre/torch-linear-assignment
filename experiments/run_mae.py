import torch 
import pandas as pd
from torch_linear_assignment import batch_linear_assignment
from itertools import product

from utils import get_current_git_branch, set_all_seeds, sum_error

set_all_seeds()


if __name__ == '__main__':

    TYPES = {
        'bf16': torch.bfloat16,
        'f16': torch.float16,
        'f32': torch.float32
    }
    FACTORS = [1, 1e1, 1e2, 1e3, 1e4, 1e5]
    NOISE_TYPES = [None, 'uniform', 'normal', 'triangular']
    NOISE_SCALES = [5e-5, 5e-4, 5e-3, 5e-1]

    BS = 100

    X = torch.load('data/train-start.pth')[:BS].to('cuda')
    A = torch.load('data/start-ans.pth')[:BS].to('cuda')

    def run(X, A, exp_type):
        res = []
        for tp, factor, noise_type, noise_scale  in product(TYPES, FACTORS,
                                                           NOISE_TYPES, NOISE_SCALES
                                                ):
            I = batch_linear_assignment(X, dtype=TYPES[tp], factor=factor,
                                         noise_scale=noise_scale, noise_type=noise_type
                                         )
            errors = sum_error(X, A, I)
            res.append(
                (tp, factor, 
                 noise_type, noise_scale, 
                 *errors)
            )

        res = pd.DataFrame(res, columns=['tp', 'factor', 
                                         'noise_type', 'noise_scale', 
                                         'error', 
                                         'mae', 'rmse', 'smape'
                                         ])
        res = res.explode('error')
        res.to_csv(f'experiments/results/errnoise_{get_current_git_branch()}_{exp_type}.csv')
    print('STARTED REAL'.center(80, '+'))
    run(X, A, 'start')

    X = torch.load('data/train-end.pth')[:BS].to('cuda')
    A = torch.load('data/end-ans.pth')[:BS].to('cuda')

    print('STARTED END'.center(80, '+'))
    run(X, A, 'end')

    X = torch.load('data/art_x.pth')[:BS].to('cuda')
    A = torch.load('data/art_ans.pth')[:BS].to('cuda')

    print('STARTED ART'.center(80, '+'))
    run(X, A, 'art')
    
