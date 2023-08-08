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

# Change size here. Should be a power of 3
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
        # Code to populate 3x3 grid
        for row in range(3):
            for col in range(3):
                if (row % 2 == 1 and col % 2 == 0) or (row % 2 == 0 and col % 2 == 1) or (row == 1 and col == 1):
                    grid[row, col] = 1

        return grid
    else:
        # recursive case
        new_length = length // 3
        # Split nxn grid into 9 (3x3) subgrids
        for row in range(3):
            for col in range(3):
                # if this ninth of the grid should be filled, call recursive case
                if (row % 2 == 1 and col % 2 == 0) or (row % 2 == 0 and col % 2 == 1) or (row == 1 and col == 1):
                    # Calculate subregion of grid as sub_grid
                    sub_row = (row * length) // 3
                    sub_col = (col * length) // 3

                    sub_grid = grid[sub_row: sub_row +
                                    new_length, sub_col: sub_col + new_length]
                    
                    # Call recursion
                    populate_grid(sub_grid, new_length)

# Start recursion
populate_grid(new_fractal, length)

# Calculate extent for centering the plot at 0,0
extent = [-1 * (length - 1) // 2, (length - 1) // 2, -
          1 * (length - 1) // 2, (length - 1) // 2]

plt.imshow(new_fractal.cpu(), cmap='gray', extent=extent)
plt.tight_layout(pad=0)
plt.show()
