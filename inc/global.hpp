#ifndef __INC_GLOBAL_HPP__


#include <assert.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>


#ifndef LAB_NUMBER
#   define LAB_NUMBER 0
#endif


enum class RCode {
    Ok           = 0
,   CUDA_Error   = 1
,   CUDA_Setup   = 2
,   TimeOut      = 3
,   MemError     = 4
,   InvalidInput = 5
,   LogicError   = 6
,   IOError      = 7
,   Unknown
};

#define CODE(RCodeValue) static_cast<int>(RCode::RCodeValue)
#define RC(Rc) RCode:: Rc

using byte = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using i32 = int32_t;


template < typename T, size_t N >
static constexpr size_t lengthof(T (&a)[N]) {
    return N;
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
