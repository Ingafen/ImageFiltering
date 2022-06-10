import numpy as np
from cuda import cuda, nvrtc

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


cuda_code = ""
with open("magic.cu") as f:
    lines = f.readlines()

cuda_code = cuda_code.join(lines)

# Create program
err, prog = nvrtc.nvrtcCreateProgram(str.encode(cuda_code), b"saxpy.cu", 0, [], [])

# Compile program
opts = []
err, = nvrtc.nvrtcCompileProgram(prog, 0, opts)

# Get PTX from compilation
err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
ptx = b" " * ptxSize
err, = nvrtc.nvrtcGetPTX(prog, ptx)
# Initialize CUDA Driver API
err, = cuda.cuInit(0)
# Retrieve handle for device 0
err, cuDevice = cuda.cuDeviceGet(0)
# Create context
err, context = cuda.cuCtxCreate(0, cuDevice)
# Load PTX as module data and retrieve function
ptx = np.char.array(ptx)
# Note: Incompatible --gpu-architecture would be detected here
err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
ASSERT_DRV(err)
err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
ASSERT_DRV(err)

NUM_THREADS = 10  # Threads per block
NUM_BLOCKS = 10  # Blocks per grid

a = np.array([2.0], dtype=np.float32)
n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
bufferSize = n * a.itemsize

hX = np.random.rand(n).astype(dtype=np.float32)
hY = np.random.rand(n).astype(dtype=np.float32)
hOut = np.zeros(n).astype(dtype=np.float32)

err, dXclass = cuda.cuMemAlloc(bufferSize)
err, dYclass = cuda.cuMemAlloc(bufferSize)
err, dOutclass = cuda.cuMemAlloc(bufferSize)

err, stream = cuda.cuStreamCreate(0)

err, = cuda.cuMemcpyHtoDAsync(
   dXclass, hX.ctypes.data, bufferSize, stream
)
err, = cuda.cuMemcpyHtoDAsync(
   dYclass, hY.ctypes.data, bufferSize, stream
)

# The following code example is not intuitive 
# Subject to change in a future release
dX = np.array([int(dXclass)], dtype=np.uint64)
dY = np.array([int(dYclass)], dtype=np.uint64)
dOut = np.array([int(dOutclass)], dtype=np.uint64)

args = [a, dX, dY, dOut, n]
args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

err, = cuda.cuLaunchKernel(
   kernel,
   NUM_BLOCKS,  # grid x dim
   1,  # grid y dim
   1,  # grid z dim
   NUM_THREADS,  # block x dim
   1,  # block y dim
   1,  # block z dim
   0,  # dynamic shared memory
   stream,  # stream
   args.ctypes.data,  # kernel arguments
   0,  # extra (ignore)
)

err, = cuda.cuMemcpyDtoHAsync(
   hOut.ctypes.data, dOutclass, bufferSize, stream
)
err, = cuda.cuStreamSynchronize(stream)
# Assert values are same after running kernel
hZ = a * hX + hY
if not np.allclose(hOut, hZ):
   raise ValueError("Error outside tolerance for host-device vectors")
else: 
    print("Сработало!")

err, = cuda.cuStreamDestroy(stream)
err, = cuda.cuMemFree(dXclass)
err, = cuda.cuMemFree(dYclass)
err, = cuda.cuMemFree(dOutclass)
err, = cuda.cuModuleUnload(module)
err, = cuda.cuCtxDestroy(context)