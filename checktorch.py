import torch
print(torch.__version__)
print("Is MPS available:",  torch.backends.mps.is_available())