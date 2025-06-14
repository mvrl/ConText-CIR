from pynvml import *
import torch

import torch.nn.functional as F

def get_most_free_device():
    if not torch.cuda.is_available():
        print("No GPU, using CPU")
        return torch.device("cpu")
    memory = []
    nvmlInit()
    for i in range(torch.cuda.device_count()):
        info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i))
        memory.append(info.free / 1024 ** 3)
    index = memory.index(max(memory))
    if memory[index] < 5:
        print("No GPU with more than 5GB of free memory, using CPU")
        return torch.device("cpu")
    else:
        print(f"Using GPU:{index} with {memory[index]}GB of free memory")
        return torch.device(f"cuda:{index}")