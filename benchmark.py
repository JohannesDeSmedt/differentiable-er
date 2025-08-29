import torch
import time

# Check for MPS and set device
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("MPS device is available. Using MPS.")
else:
    print("MPS device not found. Using CPU.")
    mps_device = torch.device("cpu")

# Function to run a benchmark
def run_benchmark(device, num_iterations=100):
    start_time = time.time()
    # Create tensors on the specified device
    x = torch.rand(1000, 1000, device=device)
    y = torch.rand(1000, 1000, device=device)
    for _ in range(num_iterations):
        z = x @ y # Matrix multiplication
    end_time = time.time()
    return end_time - start_time

# Run on CPU
cpu_time = run_benchmark(torch.device("cpu"))
print(f"CPU time: {cpu_time:.4f} seconds")

# Run on MPS
mps_time = run_benchmark(mps_device)
print(f"MPS time: {mps_time:.4f} seconds")