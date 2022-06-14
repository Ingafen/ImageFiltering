#define C_IDX blockIdx.x * blockDim.x + threadIdx.x

extern "C" __global__ void saxpy(float a, float *x, float *y, float *out, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) 
  {
    out[tid] = a * x[tid] + y[tid];
  }
}

extern "C" __global__ void learn(float* x, float* w, float* l, float* y, float* sp, 
  int* ida, int* bna, int* bca, int* s4, int* s3, int* row, int* sizes, 
  float* yExp, float* phi, float* speed)
{
  
  //Вычисление вектора y - скалярное произведение, вычисление путём редукции
  ida[C_IDX] = blockIdx.x * blockDim.x + threadIdx.x;

  while (ida[C_IDX] < sizes[0] * sizes[1]) //idx < k*M
  {
    sp[ida[C_IDX] % sizes[0] + (ida[C_IDX] / sizes[0]) * sizes[5]] = 
      x[ida[C_IDX] % sizes[0]] * w[ida[C_IDX]];
    
    ida[C_IDX] += blockDim.x * gridDim.x;
  }
  
  __syncthreads();
   
  ida[C_IDX] = blockIdx.x * blockDim.x + threadIdx.x;

  __syncthreads();

  while(s3[C_IDX] <= sizes[2]) //3-idx, 2-steps count
  {    
    __syncthreads();

    for (row[C_IDX] = 0; row[C_IDX] < sizes[1]; row[C_IDX]++)
    {
      ida[C_IDX] = blockIdx.x * blockDim.x + threadIdx.x;

      __syncthreads();

      while (ida[C_IDX] < s4[C_IDX])  
      {
        bca[C_IDX] = (1 - s3[C_IDX] % 2 ) * sizes[5] * sizes[1];
        bna[C_IDX] = (s3[C_IDX] % 2) * sizes[5] * sizes[1];

        sp[row[C_IDX] * sizes[5] + ida[C_IDX] + bca[C_IDX]] = 
          sp[row[C_IDX] * sizes[5] + 2 * ida[C_IDX] + 1 + bna[C_IDX]] 
          + sp[row[C_IDX] * sizes[5] + 2 * ida[C_IDX] + bna[C_IDX]];

        __syncthreads();   
        sp[row[C_IDX] * sizes[5] + 2*ida[C_IDX] + 1 + bna[C_IDX]] = 0;
        sp[row[C_IDX] * sizes[5] + 2*ida[C_IDX] + bna[C_IDX]] = 0;            
       
        __syncthreads();
        ida[C_IDX] += blockDim.x * gridDim.x;

        __syncthreads();
      }
    }
	 
    __syncthreads();
	
    s3[C_IDX]++;
    s4[C_IDX]/=2;
 
    __syncthreads();
  }

  ida[C_IDX] = blockIdx.x * blockDim.x + threadIdx.x;
  
  while(ida[C_IDX] < sizes[1])
  {
    y[ida[C_IDX]] = sp[ida[C_IDX] * sizes[5] + (s3[C_IDX] % 2) * sizes[5] * sizes[1]];
    ida[C_IDX] += blockDim.x * gridDim.x;
    
    __syncthreads();
  }

  __syncthreads();

  
  //Вычисление exp(y) и sum(exp(y))
  ida[C_IDX] = blockIdx.x * blockDim.x + threadIdx.x;
  while(ida[C_IDX] < sizes[1])
  {
    yExp[ida[C_IDX]] = expf(y[ida[C_IDX]]);
    atomicAdd(&yExp[sizes[1]], yExp[ida[C_IDX]]);
    ida[C_IDX] += blockDim.x * gridDim.x;
  }
  __syncthreads();

  //Вычисление phi(y)
  ida[C_IDX] = blockIdx.x * blockDim.x + threadIdx.x;
  while(ida[C_IDX] < sizes[1])
  {
    phi[ida[C_IDX]] = expf(y[ida[C_IDX]]) / yExp[sizes[1]];   
    ida[C_IDX] += blockDim.x * gridDim.x;
  }
  __syncthreads();

  
  //корректировка весов
  ida[C_IDX] = blockIdx.x * blockDim.x + threadIdx.x;

  while ( ida[C_IDX] < sizes[0] * sizes[1] )
  {
    w[ida[C_IDX]] -= speed[0]
      * x[ ida[C_IDX] % sizes[0] ] // x[i]
      * ( 
          ((1 - l[ida[C_IDX]/sizes[0]]) * phi[ida[C_IDX]/sizes[0]]) // i <> j | marker * phi_i 
          +
          (l[ida[C_IDX] / sizes[0]] * ( phi[ida[C_IDX]/sizes[0]] - 1 )) // i == j | marker * (phi_i-1)
        );

    ida[C_IDX] += blockDim.x * gridDim.x;
  }

  __syncthreads();
  
}