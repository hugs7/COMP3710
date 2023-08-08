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

length = 3**8

# Create x and y ranges using torch.arange
x_range = torch.arange(-1 * (length - 1) // 2, 1 +
                       (length - 1) // 2, STEP, device=device)
y_range = torch.arange(-1 * (length - 1) // 2, 1 +
                       (length - 1) // 2, STEP, device=device)

# Use torch.meshgrid to create meshgrid tensors
X, Y = torch.meshgrid(x_range, y_range)

# Create zeros tensor
new_fractal = torch.zeros((length, length), device=device)
# new_fractal = torch.zeros((length, length))
print(new_fractal)


def populate_grid(grid, length):
    if length == 3:  # base case
        # Code to populate the base case (e.g., filling specific cells)
        for row in range(3):
            for col in range(3):
                if (row % 2 == 1 and col % 2 == 0) or (row % 2 == 0 and col % 2 == 1) or (row == 1 and col == 1):
                    grid[row, col] = 1
                # else:
                #     grid[row, col] = 1

        return grid
    else:
        # recursive case
        new_length = length // 3
        for row in range(3):
            # print("Row", row)
            for col in range(3):
                # print("Col", col)

                if (row % 2 == 1 and col % 2 == 0) or (row % 2 == 0 and col % 2 == 1) or (row == 1 and col == 1):

                    sub_row = (row * length) // 3
                    sub_col = (col * length) // 3

                    sub_grid = grid[sub_row: sub_row +
                                    new_length, sub_col: sub_col + new_length]
                    populate_grid(sub_grid, new_length)
                    # Now you have a sub-grid filled with the fractal pattern, and you can place it in the main grid
                    # grid[row * new_length: (row + 1) * new_length, supercell *
                    #              new_length: (supercell + 1) * new_length] = sub_grid


populate_grid(new_fractal, length)

# Calculate extent for centering the plot at 0,0
extent = [-1 * (length - 1) // 2, (length - 1) // 2, -
          1 * (length - 1) // 2, (length - 1) // 2]

plt.imshow(new_fractal.cpu(), cmap='gray', extent=extent)
plt.tight_layout(pad=0)
plt.show()
