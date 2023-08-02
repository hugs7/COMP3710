import torch
import matplotlib.pyplot as plt

# Function to generate the Julia set
def julia_set(width, height, real_min, real_max, imag_min, imag_max, c_real, c_imag, max_iterations=100, escape_radius=2.0):
    # Create a grid of complex numbers on the GPU
    real = torch.linspace(real_min, real_max, width, device='cuda')
    imag = torch.linspace(imag_min, imag_max, height, device='cuda')
    real, imag = torch.meshgrid(real, imag)
    c = torch.complex(real, imag)

    # Initialize the output matrix with zeros (not in the set) on the GPU
    output = torch.zeros_like(c, dtype=torch.uint8, device='cuda')

    # Convert c_real and c_imag to tensors on the GPU
    c_real = torch.tensor(c_real, device='cuda')
    c_imag = torch.tensor(c_imag, device='cuda')

    # Iterate to determine points in the Julia set
    z = c.clone()
    for i in range(max_iterations):
        z = z * z + torch.complex(c_real, c_imag)
        in_set = (z.abs() <= escape_radius)
        output[in_set] = i + 1

    return output

# Parameters for the Julia set
width = 800
height = 600
real_min, real_max = -2.0, 2.0
imag_min, imag_max = -1.5, 1.5
c_real, c_imag = -0.52535, -0.52535  # Feel free to change these values to explore different Julia sets

# Generate the Julia set on the GPU
julia = julia_set(width, height, real_min, real_max, imag_min, imag_max, c_real, c_imag)

# Transfer the result back to the CPU for visualization
julia_cpu = julia.cpu()

# Display the Julia set
plt.imshow(julia_cpu, cmap='turbo', extent=(real_min, real_max, imag_min, imag_max))
plt.title("Julia Set")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.colorbar()
plt.show()
