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


int main(void) {

    std::cout << "*********** Lab " << LAB_NUMBER << " ***********" << std::endl;

    if (!setup_cuda()) {
        return CODE(CUDA_Setup);
    }

    return CODE(Ok);
}
