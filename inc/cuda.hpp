#ifndef __INC_CUDA_HPP__


#include "global.hpp"


#define CALL_CUDA(Func, ...) \
{ \
    cudaError_t errorCode = Func(__VA_ARGS__); \
    if (errorCode != cudaSuccess) { \
        std::cerr << "FAILED: " << cudaGetErrorString(errorCode) << std::endl; \
        return CODE(CUDA_Error); \
    } \
}


static bool setup_cuda() {

    int nbDevices;
    CALL_CUDA(cudaGetDeviceCount, &nbDevices);

    std::cout << "Number of CUDA devices: " << nbDevices << std::endl;

    if (0 == nbDevices) {
        std::cout << "No CUDA devices available!" << std::endl;
        return false;
    }

    CALL_CUDA(cudaSetDevice, 0);

    std::cout << "Successfully set device 0" << std::endl;

    return true;
}


/**
 * @brief An utility structure used to manage allocated CUDA arrays.
 */
template <typename T>
struct cuda_array_raii_t {
    T* arr;

    cuda_array_raii_t() noexcept : arr(nullptr) {}
    cuda_array_raii_t(T* a) noexcept : arr(a) {}
   ~cuda_array_raii_t() noexcept {
        if (arr != nullptr) {
            cudaFree(arr);
        }
    }

    void release(bool needFree = false) noexcept {
        if (needFree && arr != nullptr) {
            cudaFree(arr);
        }
        arr = nullptr;
    }
};


#define __INC_CUDA_HPP__
#endif
