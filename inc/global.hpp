#ifndef __INC_GLOBAL_HPP__


#include <atomic>
#include <chrono>
#include <memory>
#include <thread>


enum class RCode {
    Ok = 0
,   CUDA_Error = 1
,   CUDA_Setup = 2
,   TimeOut    = 3
,   MemError   = 4
};

#define CODE(RCodeValue) static_cast<int>(RCode::RCodeValue)


using byte = uint8_t;
using u32 = uint32_t;


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
 * @brief An utility class to measure elapsed time.
 *
 */
class Stopwatch {
public:
    using clock_t = std::chrono::high_resolution_clock;

    void start() { _tpStart = clock_t::now(); }

    float measure() const {
        auto tpNow = clock_t::now();
        std::chrono::duration<float> dur = (tpNow - _tpStart);
        return dur.count();
    }

private:
    clock_t::time_point _tpStart;
};


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


/**
 * @brief Worker thread structured data.
 *
 * I use multiple threads to precalculate sin function. In order to do that
 * several threads are created, each one of them calculates 'quant' of sin
 * values.
 */
struct WorkerData {
    std::unique_ptr<std::thread> threadPtr;

    unsigned int ixThread;
    unsigned int szQuant;

    std::atomic<bool> isFinished alignas(64);
};


#define __INC_GLOBAL_HPP__
#endif
