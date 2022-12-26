#include <cstdlib>
#include <cuda_runtime.h>
#include <iterator>
#include <ostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cublas_v2.h>
#include <cusparseLt.h>       // cusparseLt header
#include <thrust/device_vector.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

cublasHandle_t handle;

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
using scalarfn = scalar_t (*)(scalar_t);
using ewisefn = scalar_t (*)(scalar_t, scalar_t);


struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}



////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides




__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN YOUR SOLUTION
  size_t idx = 0, gid_ = gid;
  for (int i = shape.size - 1; i >= 0; i--) {
      idx += (gid_ % shape.data[i]) * strides.data[i];
      gid_ /= shape.data[i];
  }
  if (gid < size) {
      out[gid] = a[offset + idx];
  }
  /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out,
                                   size_t size, CudaVec shape, CudaVec strides,
                                   size_t offset) {
    size_t gid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idx = 0, gid_ = gid;
    for (int i = shape.size - 1; i >= 0; i--) {
        idx += (gid_ % shape.data[i]) * strides.data[i];
        gid_ /= shape.data[i];
    }
    if (gid < size) {
        out[offset + idx] = a[gid];
    }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN YOUR SOLUTION
  auto dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(
      a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t *out, size_t size,
                                    CudaVec shape, CudaVec strides,
                                    size_t offset) {

    size_t gid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idx = 0, gid_ = gid;
    for (int i = shape.size - 1; i >= 0; i--) {
        idx += (gid_ % shape.data[i]) * strides.data[i];
        gid_ /= shape.data[i];
    }
    if (gid < size) {
        out[offset + idx] = val;
    }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN YOUR SOLUTION
  auto dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(
      val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
  /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

/// BEGIN YOUR SOLUTION
__device__ __forceinline__ scalar_t multiple(scalar_t a, scalar_t b) {
    return a * b;
}
__device__ __forceinline__ scalar_t divide(scalar_t a, scalar_t b) {
    return a / b;
}
__device__ __forceinline__ scalar_t eq(scalar_t a, scalar_t b) {
    return a == b;
}
__device__ __forceinline__ scalar_t ge(scalar_t a, scalar_t b) {
    return a >= b;
}
__device__ __forceinline__ scalar_t max(scalar_t a, scalar_t b) {
    return fmax(a, b);
}
__device__ ewisefn d_mul = multiple;
__device__ ewisefn d_div = divide;
__device__ ewisefn d_eq = eq;
__device__ ewisefn d_ge = ge;
__device__ ewisefn d_max = fmax;
__device__ ewisefn d_pow = powf;
__device__ scalarfn d_exp = expf;
__device__ scalarfn d_tanh = tanhf;
__device__ scalarfn d_log = logf;

__global__ void EwiseOP(const scalar_t *a, scalar_t *out, size_t size,
                        scalarfn op) {
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size) {

        out[gid] = (*op)(a[gid]);
    }
}

__global__ void EwiseOP(const scalar_t *a, const scalar_t *b, scalar_t *out,
                        size_t size, ewisefn op) {
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size) {
        out[gid] = (*op)(a[gid], b[gid]);
    }
}

__global__ void ScalarOp(const scalar_t *a, scalar_t val, scalar_t *out,
                         size_t size, ewisefn op) {
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < size) {

        out[gid] = (*op)(a[gid], val);
    }
}

void EwiseMul(const CudaArray &a, const CudaArray &b, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn mul;
    cudaMemcpyFromSymbol(&mul, d_mul, sizeof(ewisefn));
    EwiseOP<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, mul);
}

void ScalarMul(const CudaArray &a, scalar_t val, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn mul;
    cudaMemcpyFromSymbol(&mul, d_mul, sizeof(ewisefn));
    ScalarOp<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, mul);
}
void EwiseDiv(const CudaArray &a, const CudaArray &b, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn div;
    cudaMemcpyFromSymbol(&div, d_div, sizeof(ewisefn));
    EwiseOP<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, div);
}
void ScalarDiv(const CudaArray &a, scalar_t val, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn div;
    cudaMemcpyFromSymbol(&div, d_div, sizeof(ewisefn));
    ScalarOp<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, div);
}

void ScalarPower(const CudaArray &a, scalar_t val, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn pow;
    cudaMemcpyFromSymbol(&pow, d_pow, sizeof(ewisefn));
    ScalarOp<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, pow);
}

void EwiseMaximum(const CudaArray &a, const CudaArray &b, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn maximum;
    cudaMemcpyFromSymbol(&maximum, d_max, sizeof(ewisefn));
    EwiseOP<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size,
                                     maximum);
}

void ScalarMaximum(const CudaArray &a, scalar_t val, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn maximum;
    cudaMemcpyFromSymbol(&maximum, d_max, sizeof(ewisefn));
    ScalarOp<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, maximum);
}

void EwiseEq(const CudaArray &a, const CudaArray &b, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn eq;
    cudaMemcpyFromSymbol(&eq, d_eq, sizeof(ewisefn));
    EwiseOP<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, eq);
}
void ScalarEq(const CudaArray &a, scalar_t val, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn eq;
    cudaMemcpyFromSymbol(&eq, d_eq, sizeof(ewisefn));
    ScalarOp<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, eq);
}

void EwiseGe(const CudaArray &a, const CudaArray &b, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn ge;
    cudaMemcpyFromSymbol(&ge, d_ge, sizeof(ewisefn));
    EwiseOP<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, ge);
}
void ScalarGe(const CudaArray &a, scalar_t val, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    ewisefn ge;
    cudaMemcpyFromSymbol(&ge, d_ge, sizeof(ewisefn));
    ScalarOp<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, ge);
}

void EwiseLog(const CudaArray &a, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    scalarfn log;
    cudaMemcpyFromSymbol(&log, d_log, sizeof(scalarfn));
    EwiseOP<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, log);
}
void EwiseExp(const CudaArray &a, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    scalarfn exp;
    cudaMemcpyFromSymbol(&exp, d_exp, sizeof(scalarfn));
    EwiseOP<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, exp);
}
void EwiseTanh(const CudaArray &a, CudaArray *out) {
    auto dim = CudaOneDim(out->size);
    scalarfn tanh;
    cudaMemcpyFromSymbol(&tanh, d_tanh, sizeof(scalarfn));
    EwiseOP<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, tanh);
}

/// END YOUR SOLUTION
void initCublas() {
    // Create the cuBLAS handle
    cublasCreate(&handle);
}

void cleanCublas() {
    // Destroy the cuBLAS handle
    cublasDestroy(handle);
}


__global__ void MatmulKernel_naive(scalar_t *a, scalar_t *b, scalar_t *out,
                                   uint32_t M, uint32_t N, uint32_t P) {
    size_t bidx = blockIdx.x, bidy = blockIdx.y, tidx = threadIdx.x,
           tidy = threadIdx.y;
    auto gidx = bidx * blockDim.x + tidx, gidy = bidy * blockDim.y + tidy;
    if (gidx >= M || gidy >= P) {
        return;
    }
    scalar_t sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += a[gidx * N + i] * b[i * P + gidy];
    }
    out[gidx * P + gidy] = sum;
}

__global__ void MatmulKernel_tile(scalar_t *a, scalar_t *b, scalar_t *out,
                                  uint32_t M, uint32_t N, uint32_t P) {
    size_t bidx = blockIdx.x, bidy = blockIdx.y, tidx = threadIdx.x,
           tidy = threadIdx.y;
    int x_range = static_cast<int>(bidx + 1) * TILE - M,
        y_range = static_cast<int>(bidy + 1) * TILE - P;
    if (x_range > 0) {
        a -= x_range * N;
        out -= x_range * P;
    }
    if (y_range > 0) {
        b -= y_range;
        out -= y_range;
    }
    a += bidx * TILE * N;
    b += bidy * TILE;
    out += (bidx * TILE) * P + (bidy * TILE);
    __shared__ scalar_t smemA[TILE][TILE], smemB[TILE][TILE];
    scalar_t accumu = 0.0f;
    for (int i = 0; i < N; i += TILE) {
        smemA[tidx][tidy] = (tidy + i < N) ? a[(tidx)*N + (tidy + i)] : 0.0f;
        smemB[tidx][tidy] = (tidx + i < N) ? b[(tidx + i) * P + tidy] : 0.0f;
        __syncthreads();
        for (int j = 0; j < TILE; j++) {
            accumu += smemA[tidx][j] * smemB[j][tidy];
        }
        __syncthreads();
    }
    out[tidx * P + tidy] = accumu;
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

int example(void) {
    // Host problem definition, row-major order
    constexpr int m     = 32; // bigger sizes may require dynamic allocations
    constexpr int n     = 32; // bigger sizes may require dynamic allocations
    constexpr int k     = 32; // bigger sizes may require dynamic allocations
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type  = CUDA_R_32F;
    auto          compute_type = CUSPARSE_COMPUTE_TF32;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 32;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
    auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
    auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
    auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
    auto     A_size         = A_height * lda * sizeof(float);
    auto     B_size         = B_height * ldb * sizeof(float);
    auto     C_size         = C_height * ldc * sizeof(float);
    float hA[m * k];
    float hB[k * n];
    float hC[m * n] = {};
    for (int i = 0; i < m * k; i++)
        hA[i] = static_cast<float>(static_cast<float>(std::rand() % 10));
    for (int i = 0; i < k * n; i++)
        hB[i] = static_cast<float>(static_cast<float>(std::rand() % 10));
    float alpha = 1.0f;
    float beta  = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    float *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size;
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )
    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correctness

    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    int is_valid;
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              &is_valid, stream) )

    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    std::cout << is_valid << std::endl;
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size;
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
                                            dA_compressed, stream) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUSPARSE( cusparseLtMatmulSearch(&handle, &plan, &alpha,
                                           dA_compressed, dB, &beta,
                                           dC, dD, d_workspace,
                                           streams, num_streams) )
    int alg_id;
    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                           &alg_id, sizeof(alg_id)) )
    int32_t splitK, splitKBuffers;
    cusparseLtSplitKMode_t splitKMode;

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K,
                                           &splitK, sizeof(splitK)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K_MODE,
                                           &splitKMode, sizeof(splitKMode)) )

    CHECK_CUSPARSE( cusparseLtMatmulAlgGetAttribute(
                                           &handle, &alg_sel,
                                           CUSPARSELT_MATMUL_SPLIT_K_BUFFERS,
                                           &splitKBuffers,
                                           sizeof(splitKBuffers)) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )

    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))

    CHECK_CUDA( cudaMalloc((void**)&d_workspace, workspace_size) )
    // Perform the matrix multiplication
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

    bool A_std_layout = (is_rowmajor != isA_transposed);
    bool B_std_layout = (is_rowmajor != isB_transposed);
    // host computation
    float hC_result[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum  = 0.0f;
            for (int k1 = 0; k1 < k; k1++) {
                auto posA = (A_std_layout) ? i * lda + k1 : i + k1 * lda;
                auto posB = (B_std_layout) ? k1 * ldb + j : k1 + j * ldb;
                sum      += static_cast<float>(hA[posA]) *  // [i][k]
                            static_cast<float>(hB[posB]);   // [k][j]
            }
            auto posC       = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            hC_result[posC] = sum;  // [i][j]
        }
    }
    // host-device comparison
    int correct = 1;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto pos          = (is_rowmajor) ? i * ldc + j : i + j * ldc;
            auto device_value = static_cast<float>(hC[pos]);
            auto host_value   = hC_result[pos];
            if (device_value != host_value) {
                // direct floating point comparison is not reliable
                std::printf("(%d, %d):\t%f vs. %f\n",
                            i, j, host_value, device_value);
                correct = 0;
                break;
            }
        }
    }
    if (correct)
        std::printf("spmma_example test PASSED\n");
    else
        std::printf("spmma_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    CHECK_CUDA( cudaFree(d_workspace) )
    return EXIT_SUCCESS;
}

template <typename type_t>
void cusparsespmma(
    type_t* dA,
    type_t* dB,
    type_t* dC,
    size_t m,
    size_t n,
    size_t k,
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE,
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE,
    float alpha = 1.0f,
    float beta = 0.0f) {

    // Host problem definition, row-major order
    auto          order = CUSPARSE_ORDER_COL;
    auto          type  = CUDA_R_32F;
    auto          compute_type = CUSPARSE_COMPUTE_TF32;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 32;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;


    //--------------------------------------------------------------------------
    // Device memory management
    float *dD, *dA_compressed;
    dD = dC;

    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )

    size_t workspace_size;
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )
    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correctness
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )

    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size;
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
                                            dA_compressed, stream) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;


    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))

    CHECK_CUDA( cudaMalloc((void**)&d_workspace, workspace_size) )
    // Perform the matrix multiplication
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(d_workspace) )

}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
    /********************* SPARSE ***************************/
    /********************* SPARSE ***************************/

  /// BEGIN YOUR SOLUTION
    // std::cout << std::endl << "Matmul in cuda" << std::endl;

    // Fill(out, 0.0f);
    // if (M < TILE || P < TILE || N < TILE) {
    //     dim3 block(TILE, TILE);
    //     dim3 grid((M - 1) / TILE + 1, (P - 1) / TILE + 1);
    //     MatmulKernel_naive<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
    // } else {
    //     dim3 block(TILE, TILE);
    //     dim3 grid((M - 1) / TILE + 1, (P - 1) / TILE + 1);
    //     MatmulKernel_tile<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
    // }
    // cudaDeviceSynchronize();
  /// END YOUR SOLUTION

    std::cout << M << ' ' << N << ' ' << P << std::endl;
    if (N % 8 == 0 && P % 8 == 0) {
        // sparse tc
        cusparsespmma<float>(b.ptr, a.ptr, out->ptr, P, M, N);
    }
    else {
        // cublas
        
        // cublasHandle_t cublas_handle;
        // cublasCreate(&cublas_handle);
        float cublas_alpha = 1.0f;
        float cublas_beta = 0.0f;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P, M, N, &cublas_alpha, b.ptr, P, a.ptr, N, &cublas_beta, out->ptr, P);
        // cublasDestroy(cublas_handle);
    }


}


////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
__global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out,
                                size_t reduce_size, size_t len) {
    size_t gid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gid >= len) {
        return;
    }
    scalar_t maxValue = a[gid * reduce_size];
    for (size_t i = gid * reduce_size; i < (gid + 1) * reduce_size; i++) {
        maxValue = max(maxValue, a[i]);
    }

    out[gid] = maxValue;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  auto dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size,
                                           out->size);
  /// END YOUR SOLUTION
}


__global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out,
                                size_t reduce_size, size_t len) {
    size_t gid = threadIdx.x + blockDim.x * blockIdx.x;
    scalar_t sum = 0.0f;
    if (gid >= len) {
        return;
    }
    for (size_t i = gid * reduce_size; i < (gid + 1) * reduce_size; i++) {
        sum += a[i];
    }
    out[gid] = sum;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN YOUR SOLUTION
  auto dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size,
                                           out->size);
  /// END YOUR SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("init_cublas", initCublas);
  m.def("clean_cublas", cleanCublas);
  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
