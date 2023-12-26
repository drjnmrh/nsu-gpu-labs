#include <math.h>

#include <algorithm>
#include <iostream>

#include "cuda.hpp"


#ifndef M_PI
#   define M_PI 3.14159265359
#endif


/**
 * @brief This macro is used to generate code for different sine calculating
 *        kernel functions (sin, sinf, __sinf).
 */
#define DEFINE_COMPUTE_SIN_CUDA(SinFunc) \
__global__ void compute_ ## SinFunc (int n, float* arr) { \
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; \
    if (ix < n) { \
        arr[ix] = SinFunc((ix%360)*M_PI/180.0f); \
    } \
}

#define CALL_COMPUTE_SIN_CUDA(GridSize, BlockSize, SinFunc) \
    compute_ ## SinFunc <<<GridSize, BlockSize>>>


DEFINE_COMPUTE_SIN_CUDA(sin);
DEFINE_COMPUTE_SIN_CUDA(sinf);
DEFINE_COMPUTE_SIN_CUDA(__sinf);


/**
 * @brief Computes sin values using CPU and writes out into the preallocated array.
 *
 * @param dataPtr
 *  A pointer to the worker thread data. If nullptr is given, method computes
 *  values in a single-thread mode.
 * @param n
 *  Number of values to calculate. Arguments are a sequence of values from [0, 2*PI)
 * @param out
 *  An output preallocated array.
 */
static void compute_sin_on_cpu(WorkerData* dataPtr, unsigned int n, float* out) {

    if (!out) {
        return;
    }

    unsigned int ixStart;
    unsigned int ixEnd;

    if (!!dataPtr) {
        dataPtr->isFinished.store(false, std::memory_order_relaxed);

        ixStart = dataPtr->szQuant*dataPtr->ixThread;
        ixEnd = std::min(ixStart + dataPtr->szQuant, n);
    } else {
        ixStart = 0;
        ixEnd = n;
    }

    for (unsigned int i = ixStart; i < ixEnd; ++i) {
        out[i] = sin((i%360)*M_PI/180.0);
    }

    if (!!dataPtr) {
        dataPtr->isFinished.store(true, std::memory_order_release);
    }
}


/**
 * @brief Calculates mean absolute error between values in a equal-sized float arrays.
 */
static double calculate_error(unsigned int n, float* want, float* got) {

    if (!want || !got || 0 == n) {
        return -1.0;
    }

    double err = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        err += abs(static_cast<double>(want[i] - got[i]));
    }

    return err / n;
}


static void print_error(const char* const funcname, double errorValue, float dt) {

    std::cout << " - mean error for '" << funcname << "' function: " << std::scientific << errorValue << " (" << std::fixed << dt*1000 << "ms)" << std::endl;
}


int main(void) {

    std::cout << "*********** Lab " << LAB_NUMBER << " ***********" << std::endl;

    static constexpr unsigned int N = 1e8;
    static constexpr unsigned int ArraySizeInBytes = sizeof(float)*N;
    static constexpr unsigned int WorkersNum = 8;
    static constexpr unsigned int TimeoutInMs = 3000;

    const unsigned int Quant = ceil(N/float(WorkersNum));

    std::unique_ptr<float[]> arrCpuSin = std::make_unique<float[]>(N);

    // At first precalculate sine values using CPU (multiple threads).

    WorkerData wd[WorkersNum];
    for (unsigned int ixThread = 0; ixThread < WorkersNum; ++ixThread) {
        wd[ixThread].isFinished.store(false, std::memory_order_relaxed);
        wd[ixThread].ixThread = ixThread;
        wd[ixThread].szQuant = Quant;
        wd[ixThread].threadPtr =
            std::make_unique<std::thread>( compute_sin_on_cpu, &wd[ixThread]
                                         , N, arrCpuSin.get());
    }

    Stopwatch stopwatch;
    stopwatch.start();
    while(true) {
        bool hasFinished = true;
        for (unsigned int it = 0; it < WorkersNum; ++it) {
            hasFinished = (hasFinished && wd[it].isFinished.load(std::memory_order_acquire));
        }
        if (hasFinished) {
            break;
        }

        if (stopwatch.measure() * 1000.0f > TimeoutInMs) {
            std::cerr << "TIMEOUT!" << std::endl;
            for (unsigned int it = 0; it < WorkersNum; ++it) {
                if (wd[it].threadPtr->joinable())
                    wd[it].threadPtr->detach();
            }
            arrCpuSin.release();
            return CODE(TimeOut);
        }
    }

    std::cout << "Finished calculating in " << (stopwatch.measure() * 1000.0f) << "ms" << std::endl;

    for (unsigned int it = 0; it < WorkersNum; ++it) {
        if (wd[it].threadPtr->joinable())
            wd[it].threadPtr->join();
    }

    // Calculate sine values using different CUDA sine functions and calculate
    // error (precalculated CPU values are used as a ground truth).

    if (!setup_cuda()) {
        return CODE(CUDA_Setup);
    }

    float* gpuArray;

    CALL_CUDA(cudaMalloc, &gpuArray, ArraySizeInBytes);
    cuda_array_raii_t<float> gpuArrayRaii(gpuArray);

    std::unique_ptr<float[]> cpuArray = std::make_unique<float[]>(N);

    dim3 BlockSz(512);
    dim3 GridSz(ceil(N/float(BlockSz.x)));

    float dt;

    stopwatch.start();
    CALL_COMPUTE_SIN_CUDA(GridSz, BlockSz, sin)(N, gpuArray);

    CALL_CUDA(cudaPeekAtLastError);
    CALL_CUDA(cudaDeviceSynchronize);
    dt = stopwatch.measure();

    CALL_CUDA(cudaMemcpy, cpuArray.get(), gpuArray, ArraySizeInBytes, cudaMemcpyDeviceToHost);
    double meanError = calculate_error(N, arrCpuSin.get(), cpuArray.get());
    print_error("sin", meanError, dt);

    stopwatch.start();
    CALL_COMPUTE_SIN_CUDA(GridSz, BlockSz, sinf)(N, gpuArray);

    CALL_CUDA(cudaPeekAtLastError);
    CALL_CUDA(cudaDeviceSynchronize);
    dt = stopwatch.measure();

    CALL_CUDA(cudaMemcpy, cpuArray.get(), gpuArray, ArraySizeInBytes, cudaMemcpyDeviceToHost);
    meanError = calculate_error(N, arrCpuSin.get(), cpuArray.get());
    print_error("sinf", meanError, dt);

    stopwatch.start();
    CALL_COMPUTE_SIN_CUDA(GridSz, BlockSz, __sinf)(N, gpuArray);

    CALL_CUDA(cudaPeekAtLastError);
    CALL_CUDA(cudaDeviceSynchronize);
    dt = stopwatch.measure();

    CALL_CUDA(cudaMemcpy, cpuArray.get(), gpuArray, ArraySizeInBytes, cudaMemcpyDeviceToHost);
    meanError = calculate_error(N, arrCpuSin.get(), cpuArray.get());
    print_error("__sinf", meanError, dt);

    return CODE(Ok);
}
