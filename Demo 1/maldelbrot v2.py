# This version doesn't use numpy which is faster becasue it doesn't need to
# copy data from system memory to the GPU

import torch
import matplotlib.pyplot as plt

print("PyTorch Version: " + torch.__version__)

device = torch.device('cuda')
print(torch.cuda.is_available())

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

Y, X = torch.meshgrid(torch.arange(-1.3, 1.3, 0.002),
                      torch.arange(-2, 1, 0.002))
x = X.to(device)
y = Y.to(device)

z = torch.complex(x, y)
zs = z
ns = torch.zeros_like(z)

z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

for i in range(2000):
    zs_ = zs * zs + z
    not_div = torch.abs(zs_) < 4.0
    ns += not_div.to(torch.float32)
    zs = zs_

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

fig = plt.figure(figsize=(16, 10))


def procFrac(a):
    a_cyclic = (6.28 * a / 20.0).reshape((*a.shape, 1))
    img = torch.cat([10 + 20 * torch.cos(a_cyclic),
                     30 + 50 * torch.sin(a_cyclic),
                     155 - 80 * torch.cos(a_cyclic)], 2)
    img_max = torch.max(img)
    img[img == img_max] = 0
    a = img
    a = a.to(torch.uint8)  # Corrected line
    return a


plt.imshow(procFrac(ns.abs().cpu()).numpy())
plt.tight_layout(pad=0)
plt.show()
