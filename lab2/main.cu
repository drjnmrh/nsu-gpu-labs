#include <math.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include <png.h>

#include "cuda.hpp"

// Too much hassle to setup Makefile for linkage with a separate static lib -
// using unity build idea here.
#include "assets.cpp"


static constexpr u32 cKernelSize = 3;
static constexpr float cKernel[] = {
    1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
    2.0f/16.0f, 4.0f/16.0f, 2.0f/16.0f,
    1.0f/16.0f, 2.0f/16.0f, 1.0f/16.0f,
};

__constant__ static float cDeviceKernel[lengthof(cKernel)];


__global__ static
void convolute_in_global_mem(byte* dst, const byte* src, u32 width, u32 height, u32 stride, u32 channels) {

    u32 col = threadIdx.x + blockDim.x*blockIdx.x;
    u32 row = threadIdx.y + blockDim.y*blockIdx.y;

    if (col >= width || row >= height) {
        return;
    }

    for (u32 ch = 0; ch < channels; ++ch) {
        u32 ix = stride*row + col*channels + ch;
        dst[ix] = 0;
        for (u32 i = 0; i < cKernelSize; ++i) {
            for (u32 j= 0; j < cKernelSize; ++j) {
                i32 c = col + j - 1;
                i32 r = row + i - 1;
                if (c < 0 || r < 0 || c >= width || r >= height) {
                    continue;
                }
                dst[ix] += src[stride*r + c*channels+ch]*cDeviceKernel[cKernelSize*i+j];
            }
        }
    }
}


int main(int argc, char** argv) {

    std::cout << "*********** Lab " << LAB_NUMBER << " ***********" << std::endl;

    if (!setup_cuda()) {
        return CODE(CUDA_Setup);
    }

    AssetsManager am;
    RCode rc = am.Setup();
    if (rc != RCode::Ok) {
        std::cerr << "Failed to setup Assets Manager (" << (int)rc << ")" << std::endl;
        return static_cast<int>(rc);
    }

    AssetData ad;
    rc = am.Load(ad, "stonedfox_artsy.png");
    if (rc != RCode::Ok) {
        std::cerr << "Failed to load asset (" << (int)rc << ")" << std::endl;
        return static_cast<int>(rc);
    }

    Png png;
    rc = png.Load(ad);
    if (rc != RCode::Ok) {
        std::cerr << "Failed to load PNG (" << (int)rc << ")" << std::endl;
        return static_cast<int>(rc);
    }
    rc = png.Convert(Png::ColorFormat::R8G8B8);
    if (rc != RCode::Ok) {
        std::cerr << "Failed to convert PNG (" << (int)rc << ")" << std::endl;
        return static_cast<int>(rc);
    }

    CALL_CUDA(cudaMemcpyToSymbol, cDeviceKernel, cKernel, sizeof(cKernel));

    byte* source;
    CALL_CUDA(cudaMalloc, &source, png.Height()*png.RowSize());

    return CODE(Ok);
}
