import torch
import numpy as np
import matplotlib.pyplot as plt

print("PyTorch Version: " + torch.__version__)

device = torch.device('cuda')
print(torch.cuda.is_available())

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
x = torch.Tensor(X)
y = torch.Tensor(Y)

z = torch.complex(x, y)
zs = z
ns = torch.zeros_like(z)
c = (-0.5 + -0.5j)

z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

for i in range(2000):
    zs_ = zs * zs + c
    not_div = torch.abs(zs_) < 4.0
    ns += not_div.type(ns.dtype)
    zs = zs_


# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')


fig = plt.figure(figsize=(16, 10))


def procFrac(a):
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10+20*np.cos(a_cyclic), 30+50 *
                         np.sin(a_cyclic), 155-80*np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a


plt.imshow(procFrac(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()
