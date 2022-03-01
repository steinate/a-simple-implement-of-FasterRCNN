"""
tools to convert specified type
"""
import torch as t
import numpy as np
from config import device


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def totensor(data):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        # detach() 返回相同数据的 tensor ,且 requires_grad=False
        tensor = data.detach()
    return tensor.to(device)


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()