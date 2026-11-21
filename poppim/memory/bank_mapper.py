import numpy as np


def map_banks(context_length, num_banks, policy='striped'):
    if context_length < 0 or num_banks < 1:
        raise ValueError('invalid context length or bank count')
    ids = np.arange(context_length, dtype=np.int64)
    if policy == 'striped':
        return ids % num_banks
    if policy == 'contiguous':
        return np.minimum(ids*num_banks//max(context_length, 1), num_banks-1)
    raise ValueError('mapping must be striped or contiguous')
