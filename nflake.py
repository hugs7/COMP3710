import torch
import matplotlib.pyplot as plt

# Function to generate the next level of the n-flake
def n_flake_iteration(points):
    num_points = points.shape[0]
    new_points = torch.zeros((num_points * 2 - 1, 2), dtype=torch.float32)

    # Calculate intermediate points
    for i in range(num_points - 1):
        p1, p2 = points[i], points[i + 1]
        q = p1 + (p2 - p1) / 3
        s = p1 + (p2 - p1) * 2 / 3
        r = torch.tensor([q[0] + (s[0] - q[0]) * 0.5 - (s[1] - q[1]) * (3**0.5) * 0.5,
                          q[1] + (s[1] - q[1]) * 0.5 + (s[0] - q[0]) * (3**0.5) * 0.5])
        new_points[2 * i] = p1
        new_points[2 * i + 1] = r
        new_points[2 * i + 2] = s

    # Copy the last point to complete the loop
    new_points[-1] = points[-1]

    return new_points

# Function to generate the n-flake fractal
def n_flake(n, iterations=4):
    # Initial triangle points
    points = torch.tensor([[0, 0], [1, 0], [0.5, (3**0.5) / 2]], dtype=torch.float32)

    for _ in range(iterations):
        points = n_flake_iteration(points)

    return points

# Plot the n-flake fractal
n = 5  # Change the value of n to control the complexity of the fractal
points = n_flake(n)
x, y = points[:, 0], points[:, 1]
x = torch.cat([x, x[0:1]])  # To connect the last point to the first point
y = torch.cat([y, y[0:1]])  # To connect the last point to the first point
plt.plot(x.numpy(), y.numpy())
plt.title(f"{n}-Flake Fractal")
plt.axis("equal")
plt.show()
