import torch

import torch_linear_assignment._backend as backend
from scipy.optimize import linear_sum_assignment


def batch_linear_assignment_cpu(cost):
    b, w, t = cost.shape
    matching = torch.full([b, w], -1, dtype=torch.long, device=cost.device)
    for i in range(b):
        workers, tasks = linear_sum_assignment(cost[i].numpy(), maximize=False)  # (N, 2).
        workers = torch.from_numpy(workers)
        tasks = torch.from_numpy(tasks)
        matching[i].scatter_(0, workers, tasks)
    return matching


def batch_linear_assignment_cuda(cost, dtype=None, scale_factor=1, clamp_factor=.9):
    if dtype is None:
        dtype = cost.dtype
    b, w, t = cost.shape
    # raise Exception(str(cost.dtype))
    if t < w:
        cost = cost.transpose(1, 2)
    if dtype in (torch.float16, torch.bfloat16):
        cost = _whiten_data(cost, scale_factor=scale_factor).to(dtype)
    maxval = torch.finfo(dtype).max * clamp_factor
    torch.clamp_(cost, -maxval, maxval)
    if dtype not in (torch.float16, torch.bfloat16):
        result = backend.batch_linear_assignment(cost.contiguous().float())
    elif dtype is torch.bfloat16:
        result = backend.bla_bf16(cost.contiguous())
    else:
        result = backend.batch_linear_assignment_half(cost.contiguous())
    ret = (result[-1] if t < w else result[0]).long()
    return ret
    
from torch.nn.functional import tanh  
def _whiten_data(costs, **kws):
    costs -= torch.mean(costs, dim=(1, 2), keepdim=True)
    costs /= torch.std(costs, dim=(1, 2), keepdim=True)
    if 'tanh_range' in kws:
        costs = tanh(kws['tanh_range'] * costs) * costs
    if 'scale_factor' in kws:
        scale_factor = kws['scale_factor']
        if scale_factor != 1:
            costs = costs * scale_factor
    return costs


def batch_linear_assignment(cost, dtype=None, factor=1, **kwargs):
    """Solve a batch of linear assignment problems.

    The method minimizes the cost.

    Args:
      cost: Cost matrix with shape (B, W, T), where W is the number of workers
            and T is the number of tasks.

    Returns:
      Matching tensor with shape (B, W), with assignments for each worker. If the
      task was not assigned, the corresponding index will be -1.
    """
    if cost.ndim != 3:
        raise ValueError("Need 3-dimensional tensor with shape (B, W, T).")

    if backend.has_cuda() and cost.is_cuda:
        if cost.dtype in (torch.long, torch.int, torch.int16, torch.int8):
            cost = cost.to(torch.float32)
        return batch_linear_assignment_cuda(cost, dtype, scale_factor=factor)
    else:
        return batch_linear_assignment_cpu(cost)


def assignment_to_indices(assignment):
    """Convert assignment to the SciPy format.

    Args:
        assignment: The assignment with shape (B, W).

    Returns:
        row_ind, col_ind: An array of row indices and one of corresponding column indices
            giving the optimal assignment, each with shape (B, K).

    Raises:
        ValueError if batch assignments have different sizes.
    """
    batch_size = assignment.shape[0]
    if batch_size == 0:
        indices = torch.zeros(0, 0, dtype=torch.long, device=assignment.device)
        return indices, indices
    mask = assignment >= 0
    n_matches = mask.sum(1)
    if (n_matches[1:] != n_matches[0]).any():
        raise ValueError("Inconsistent matching sizes.")
    n_matches = n_matches[0].item()
    row_ind = mask.nonzero()[:, 1].reshape(batch_size, n_matches)
    col_ind = assignment.masked_select(mask).reshape(batch_size, n_matches)
    return row_ind, col_ind
