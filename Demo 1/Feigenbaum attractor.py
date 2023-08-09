import torch
import matplotlib.pyplot as plt

print("PyTorch Version: " + torch.__version__)

device = torch.device('cuda')
print(torch.cuda.is_available())

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# R, X = torch.meshgrid(torch.arange(0, 5, 0.01),
#                       torch.arange(0, 1, 0.001))
R, X = torch.meshgrid(torch.arange(0, 5, 0.01),
                      torch.arange(0, 1, 0.01))
x = X.to(device)
r = R.to(device)

# plt.plot(x.abs().cpu().numpy(), 'b.')
# plt.show()

# plt.plot(r.abs().cpu().numpy(), 'b.')
# plt.show()

for i in range(2000):
    x = r * x * (1 - x)

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')

fig = plt.figure(figsize=(16, 10))


plt.plot(r.abs().cpu().numpy(),x.abs().cpu().numpy(), 'b.')
plt.tight_layout(pad=0)
plt.show()
