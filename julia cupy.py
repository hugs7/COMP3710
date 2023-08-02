import cupy as cp
import matplotlib.pyplot as plt

# Function to generate the Julia set
def julia_set(width, height, real_min, real_max, imag_min, imag_max, c_real, c_imag, max_iterations=100, escape_radius=2.0):
    # Rest of the code remains the same...
    # Instead of using torch, we now use cupy

    # Create a grid of complex numbers on the GPU
    real = cp.linspace(real_min, real_max, width)
    imag = cp.linspace(imag_min, imag_max, height)
    real, imag = cp.meshgrid(real, imag)
    c = cp.complex(real, imag)

    # Initialize the output matrix with zeros (not in the set) on the GPU
    output = cp.zeros_like(c, dtype=cp.uint8)

    # Convert c_real and c_imag to tensors on the GPU
    c_real = cp.array(c_real)
    c_imag = cp.array(c_imag)

    # Iterate to determine points in the Julia set
    z = c.copy()
    for i in range(max_iterations):
        z = z * z + cp.complex(c_real, c_imag)
        in_set = (z.abs() <= escape_radius)
        output[in_set] = i + 1

    return output

# Rest of the code remains the same...

# Transfer the result back to the CPU for visualization
julia_cpu = cp.asnumpy(julia)

# Display the Julia set
plt.imshow(julia_cpu, cmap='hot', extent=(real_min, real_max, imag_min, imag_max))
plt.title("Julia Set")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.colorbar()
plt.show()
