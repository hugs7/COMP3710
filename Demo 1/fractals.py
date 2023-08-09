import torch
import math
import numpy as np

print("PyTorch Version: " + torch.__version__)

device = torch.device('cuda')
print(torch.cuda.is_available())
X,Y = np.mgrid[-40.0:40:0.01, -40.0:40:0.01]

x = torch.Tensor(X)
x = x.to(device)
y = torch.Tensor(Y)
y = y.to(device)

z = torch.sin((x*np.sin((math.pi / 180) * 90) + y * np.cos((math.pi / 180) * 90)) /200.0)
z_cpu = z.cpu().numpy()
import matplotlib.pyplot as plt

plt.imshow(z_cpu)

plt.tight_layout()
plt.show()