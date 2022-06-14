from math import ceil, log2
import cv2
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

def read_learn_sample(d):
    img = cv2.imread('.\IR\l1.jpg', cv2.IMREAD_GRAYSCALE)
    scale = 3
    im_width = int(img.shape[1] / scale)
    im_height = int(img.shape[0] / scale)
    im_size = (im_width, im_height)
    learnDataCount = 21
    classes_count = 21

    learn = np.zeros((learnDataCount, im_height * im_width), dtype=np.float32)
    answer = np.zeros((learnDataCount, classes_count), dtype=np.float32)

    for i in range(0,learnDataCount):
        tmp_img = cv2.imread(".\IR\l{}.jpg".format(str(i+1)), cv2.IMREAD_GRAYSCALE)
        tmp_img = cv2.resize(tmp_img, im_size)
        tmp = tmp_img.flatten()
        learn[i,:] = tmp * d
        answer[i,i] = 1.0
    return learn, answer
    
def arr_size(x):
    debugg = x.size * x.itemsize
    return  debugg

def cuda_learn(x: np.array, l: np.array, w: np.array):
    cuda_code = ""
    with open("magic.cu") as f:
        lines = f.readlines()

    cuda_code = cuda_code.join(lines)
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(cuda_code), b"saxpy.cu", 0, [], [])
    opts = []
    err, = nvrtc.nvrtcCompileProgram(prog, 0, opts)
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    err, = cuda.cuInit(0)
    err, cuDevice = cuda.cuDeviceGet(0)
    err, context = cuda.cuCtxCreate(0, cuDevice)
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, b"learn")
    ASSERT_DRV(err)

    NUM_THREADS = 31  # Threads per block
    NUM_BLOCKS = 1 # Blocks per grid

    M = x.shape[0] # height * width
    k = l.shape[0] # classes count
    #w = np.zeros((k,M), dtype=np.float32) # neurons weight
    y = np.zeros(k, dtype=np.float32) # inner product of weights and input vector
    sp_hor_size = pow(2, ceil(log2(M)))
    sp = np.zeros((2 * k * sp_hor_size), dtype=np.float32) # place for inner product and reduction
    ida = np.zeros(NUM_BLOCKS * NUM_THREADS, dtype=np.int32) # place for cuda idx
    bna = np.zeros(NUM_BLOCKS * NUM_THREADS, dtype=np.int32) # place for bias_nechet
    bca = np.zeros(NUM_BLOCKS * NUM_THREADS, dtype=np.int32) # place for bias_chet

    sizes = np.zeros(12, dtype=np.int32) # place for indexes
    sizes[0] = M
    sizes[1] = k
    sizes[2] = ceil(log2(M))-1 # reduction steps count
    sizes[3] = 0 # reduction step idx
    sizes[4] = sp_hor_size / 2 # reduction step limit
    sizes[5] = sp_hor_size   
    s4 = np.full(NUM_BLOCKS * NUM_THREADS, sizes[4], dtype=np.int32) #sizes[2]
    s3 = np.full(NUM_BLOCKS * NUM_THREADS, sizes[3], dtype=np.int32) #sizes[3]
    row = np.zeros(NUM_BLOCKS * NUM_THREADS, dtype=np.int32) #memory for for loop
    yExp = np.zeros(sizes[1] + 1, dtype=np.float32) #memory for exp(y), additional element - for sum(exp(y))
    phi = np.zeros(sizes[1], dtype=np.float32) # for phy(y)
    speed = np.full(1, 0.5, dtype=np.float32) # learning speed rate

    err, stream = cuda.cuStreamCreate(0)

    err, d_x_class = cuda.cuMemAlloc(arr_size(x))
    err, = cuda.cuMemcpyHtoDAsync(d_x_class, x.ctypes.data, arr_size(x), stream)
    err, d_l_class = cuda.cuMemAlloc(arr_size(l))
    err, = cuda.cuMemcpyHtoDAsync(d_l_class, l.ctypes.data, arr_size(l), stream)
    err, d_w_class = cuda.cuMemAlloc(arr_size(w))
    err, = cuda.cuMemcpyHtoDAsync(d_w_class, w.ctypes.data, arr_size(w), stream)
    err, d_y_class = cuda.cuMemAlloc(arr_size(y))
    err, = cuda.cuMemcpyHtoDAsync(d_y_class, y.ctypes.data, arr_size(y), stream)
    err, d_sp_class = cuda.cuMemAlloc(arr_size(sp))
    err, = cuda.cuMemcpyHtoDAsync(d_sp_class, sp.ctypes.data, arr_size(sp), stream)
    err, d_ida_class = cuda.cuMemAlloc(arr_size(ida))
    err, = cuda.cuMemcpyHtoDAsync(d_ida_class, ida.ctypes.data, arr_size(ida), stream)
    err, d_bna_class = cuda.cuMemAlloc(arr_size(bna))
    err, = cuda.cuMemcpyHtoDAsync(d_bna_class, bna.ctypes.data, arr_size(bna), stream)
    err, d_bca_class = cuda.cuMemAlloc(arr_size(bca))
    err, = cuda.cuMemcpyHtoDAsync(d_bca_class, bca.ctypes.data, arr_size(bca), stream)
    err, d_s4_class = cuda.cuMemAlloc(arr_size(s4))
    err, = cuda.cuMemcpyHtoDAsync(d_s4_class, s4.ctypes.data, arr_size(s4), stream)
    err, d_s3_class = cuda.cuMemAlloc(arr_size(s3))
    err, = cuda.cuMemcpyHtoDAsync(d_s3_class, s3.ctypes.data, arr_size(s3), stream)
    err, d_row_class = cuda.cuMemAlloc(arr_size(row))
    err, = cuda.cuMemcpyHtoDAsync(d_row_class, row.ctypes.data, arr_size(row), stream)
    err, d_sizes_class = cuda.cuMemAlloc(arr_size(sizes))
    err, = cuda.cuMemcpyHtoDAsync(d_sizes_class, sizes.ctypes.data, arr_size(sizes), stream)
    err, d_yExp_class = cuda.cuMemAlloc(arr_size(yExp))
    err, = cuda.cuMemcpyHtoDAsync(d_yExp_class, yExp.ctypes.data, arr_size(yExp), stream)
    err, d_phi_class = cuda.cuMemAlloc(arr_size(phi))
    err, = cuda.cuMemcpyHtoDAsync(d_phi_class, phi.ctypes.data, arr_size(phi), stream)
    err, d_speed_class = cuda.cuMemAlloc(arr_size(speed))
    err, = cuda.cuMemcpyHtoDAsync(d_speed_class, speed.ctypes.data, arr_size(speed), stream)

    d_x = np.array([int(d_x_class)], dtype=np.uint64)
    d_l = np.array([int(d_l_class)], dtype=np.uint64)
    d_w = np.array([int(d_w_class)], dtype=np.uint64)
    d_y = np.array([int(d_y_class)], dtype=np.uint64)
    d_sp = np.array([int(d_sp_class)], dtype=np.uint64)
    d_ida = np.array([int(d_ida_class)], dtype=np.uint64)
    d_bna = np.array([int(d_bna_class)], dtype=np.uint64)
    d_bca = np.array([int(d_bca_class)], dtype=np.uint64)
    d_s4 = np.array([int(d_s4_class)], dtype=np.uint64)
    d_s3 = np.array([int(d_s3_class)], dtype=np.uint64)
    d_row = np.array([int(d_row_class)], dtype=np.uint64)
    d_sizes = np.array([int(d_sizes_class)], dtype=np.uint64)
    d_yExp = np.array([int(d_yExp_class)], dtype=np.uint64)
    d_phi = np.array([int(d_phi_class)], dtype=np.uint64)
    d_speed = np.array([int(d_speed_class)], dtype=np.uint64)

    args = [d_x, d_w, d_l, d_y, d_sp, d_ida, d_bna, d_bca, d_s4, d_s3, d_row, d_sizes, d_yExp, d_phi, d_speed]    
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

    err, = cuda.cuMemcpyDtoHAsync(w.ctypes.data, d_w_class, arr_size(w), stream)
    err, = cuda.cuMemcpyDtoHAsync(phi.ctypes.data, d_phi_class, arr_size(phi), stream)
    err, = cuda.cuStreamSynchronize(stream)

    print(phi)
    print("")
    
    err, = cuda.cuStreamDestroy(stream)
    err, = cuda.cuMemFree(d_x_class)
    err, = cuda.cuMemFree(d_w_class)
    err, = cuda.cuMemFree(d_l_class)
    err, = cuda.cuMemFree(d_y_class)
    err, = cuda.cuMemFree(d_sp_class)
    err, = cuda.cuMemFree(d_ida_class)
    err, = cuda.cuMemFree(d_bna_class)
    err, = cuda.cuMemFree(d_bca_class)
    err, = cuda.cuMemFree(d_s3_class)
    err, = cuda.cuMemFree(d_s4_class)
    err, = cuda.cuMemFree(d_sizes_class)
    err, = cuda.cuMemFree(d_row_class)
    err, = cuda.cuMemFree(d_sizes) 
    err, = cuda.cuMemFree(d_yExp) 
    err, = cuda.cuMemFree(d_phi) 
    err, = cuda.cuMemFree(d_speed) 
    err, = cuda.cuModuleUnload(module)
    err, = cuda.cuCtxDestroy(context)
    return w


ll = read_learn_sample(1e-4)
M = ll[0].shape[1] # height * width
k = ll[1].shape[0] # classes count
w1 = np.zeros((k * M), dtype=np.float32) # neurons weight

for n in range(0,4):
    for i in range(0, ll[0].shape[0]):
        print("Обучение образцом "+str(i) + " проход " + str(n))
        w1 = cuda_learn(x=ll[0][i], l=ll[1][i], w=w1)