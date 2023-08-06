# Square fractal

import torch
import matplotlib.pyplot as plt

print("PyTorch Version: " + torch.__version__)

device = torch.device('cuda')
print(torch.cuda.is_available())

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
STEP = 1

# torch.arange(start, stop, step) returns 1D tensor
# x_range = torch.arange(0, 1, STEP)
# y_range = torch.arange(0, 1, STEP)
x_len, y_len = 3, 3
# Torch.mesgrid (Tensor X, Tensor Y)
# X, Y = torch.meshgrid(x_range, y_range)
# print(X)
# print(Y)
# 3x3 grid to start where 0,0 is the centre
# so -1 to 1 on x and y

# Move meshgrid onto the GPU
# x = X.to(device)
# y = Y.to(device)

# Create zeros array
# grid_z = torch.zeros_like(X)
# for i, row in enumerate(grid_z):
#     for j, cell in enumerate(row):
#         if (i % 2 == 1 and j % 2 == 0) or (i % 2 == 0 and j % 2 == 1):
#             grid_z[i][j] = 2
# print(grid_z)


def growSquare(x_length, y_length):
    # Takes n*n meshgrid and makes it 3n*3n

    new_X_t = torch.arange(-3 * (x_length - 1) // 2,
                           3 * (x_length - 1) // 2, STEP)
    new_Y_t = torch.arange(-3 * (y_length - 1) // 2,
                           3 * (y_length - 1) // 2, STEP)

    # Make new grid
    new_Y, new_X = torch.meshgrid(new_X_t, new_Y_t)

    # Create a tensor to hold the fractal values
    fractal = torch.zeros_like(new_X)
    return fractal


def populate_grid(fractal_grid, length):
    if length == 3:  # base case
        # Code to populate the base case (e.g., filling specific cells)
        for i, row in enumerate(fractal_grid):
            for j, cell in enumerate(row):
                if (i % 2 == 1 and j % 2 == 0) or (i % 2 == 0 and j % 2 == 1):
                    fractal_grid[i][j] = 1
        print(fractal_grid)

        return fractal_grid
    else:
        # recursive case
        new_length = length // 3
        for row in range(3):
            for supercell in range(3):
                if (row % 2 == 1 and supercell % 2 == 0) or (row % 2 == 0 and supercell % 2 == 1):
                    sub_grid = torch.zeros(
                        new_length, new_length, dtype=fractal_grid.dtype)
                    populate_grid(sub_grid, new_length)
                    # Now you have a sub-grid filled with the fractal pattern, and you can place it in the main grid
                    fractal_grid[row * new_length: (row + 1) * new_length, supercell *
                                 new_length: (supercell + 1) * new_length] = sub_grid


for i in range(5):
    # Test the function
    new_fractal = growSquare(x_len, y_len)
    populate_grid(new_fractal, x_len)
    x_len *= 3
    y_len *= 3
    print(x_len, y_len)

plt.imshow(new_fractal.cpu(), cmap='gray')
plt.tight_layout(pad=0)
plt.show()
