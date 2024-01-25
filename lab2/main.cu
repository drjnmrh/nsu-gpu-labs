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


static constexpr u32 cBlockSize = 16;


__global__ static 
void convolute_in_shared_mem(byte* dst, const byte* src, u32 width, u32 height, u32 stride, u32 channels) {

	static constexpr u32 cSharedBlockSize = cBlockSize + cKernelSize - 1;

	__shared__ byte sharedbuf[cSharedBlockSize][cSharedBlockSize];

	u32 flatten = threadIdx.y*blockDim.x + threadIdx.x;
	u32 blockarea = blockDim.x*blockDim.y;

	i32 sx = blockIdx.x*blockDim.x - cKernelSize/2;
	i32 sy = blockIdx.y*blockDim.y - cKernelSize/2;

	for (u32 ch = 0; ch < channels; ++ch) {
		u32 dcol = flatten%cSharedBlockSize;
		u32 drow = flatten/cSharedBlockSize;
		i32 scol = sx + dcol;
		i32 srow = sy + drow;

		if (scol >= 0 && scol < width && srow >= 0 && srow < height)
			sharedbuf[drow][dcol] = src[stride*srow + scol*channels + ch];
		else
			sharedbuf[drow][dcol] = 0;

		dcol = (flatten + blockarea)%cSharedBlockSize;
		drow = (flatten + blockarea)/cSharedBlockSize;
		scol = sx + dcol;
		srow = sy + drow;

		if (scol >= 0 && scol < width && srow >= 0 && srow < height)
			sharedbuf[drow][dcol] = src[stride*srow + scol*channels + ch];
		else
			sharedbuf[drow][dcol] = 0;

		__syncthreads();

		u32 c = blockIdx.x*blockDim.x + threadIdx.x;
		u32 r = blockIdx.y*blockDim.y + threadIdx.y;
		if (c < width && r < height) {
			u32 ix = stride*r + channels*c + ch;
			float sum = 0;
			for (u32 i = 0; i < cKernelSize; ++i) {
				for (u32 j = 0; j < cKernelSize; ++j) {
					sum += sharedbuf[threadIdx.y+j][threadIdx.x+i]*cDeviceKernel[cKernelSize*j+i];
				}
			}

			dst[ix] = (byte)min(max((int)sum, 0), 255);
		}

		__syncthreads();
	}
}


static texture<byte, cudaTextureType1D, cudaReadModeElementType> deviceInputTexture;


__global__ static
void convolute_in_texture(byte* dst, const byte* src, u32 width, u32 height, u32 stride, u32 channels) {

	u32 c = blockIdx.x*blockDim.x + threadIdx.x;
	u32 r = blockIdx.y*blockDim.y + threadIdx.y;

	if (r >= height || c >= width) return;

	for (u32 ch = 0; ch < channels; ++ch) {
		u32 ix = stride*r + channels*c + ch;
		float sum = 0;

		for (u32 i = 0; i < cKernelSize; ++i) {
			for (u32 j = 0; j < cKernelSize; ++j) {
				i32 row = r + j - 1;
				i32 col = c + i - 1;

				if (row >= 0 && row < height && col >= 0 && col < width) {
					sum += tex1Dfetch(deviceInputTexture, row*stride + channels*col + ch) * cDeviceKernel[cKernelSize*j+i];
				}
			}
		}
		
		dst[ix] = (byte)min(max((int)sum, 0), 255);
	}	
}


RCode blur_with_global(AssetsManager& am, byte* source, Png& png) {

	byte* dest;
	CALL_CUDA(cudaMalloc, &dest, png.Height()*png.RowSize());
	cuda_array_raii_t _raii_dest(dest);

	dim3 szblock(cBlockSize, cBlockSize);
	dim3 szgrid(static_cast<int>(std::ceil(png.Width()/(float)cBlockSize)), static_cast<int>(std::ceil(png.Height()/(float)cBlockSize)));
	
	Stopwatch sw;
	sw.start();
	for (u32 i = 0; i < 100; ++i) {
		convolute_in_global_mem<<<szgrid, szblock>>>(dest, source, png.Width(), png.Height(), png.RowSize(), 3);

		CALL_CUDA(cudaPeekAtLastError);
		CALL_CUDA(cudaDeviceSynchronize);
	}

	float t = sw.measure()/100.0f;
	std::cout << "Global Elapsed: " << t*1000.0f << "ms" << std::endl;

	CALL_CUDA(cudaMemcpy, png.Data(), dest, png.Height()*png.RowSize(), cudaMemcpyDeviceToHost);

	AssetData ad;
	RCode rc = png.Save(ad);
	if (rc != RCode::Ok) {
		std::cerr << "Failed to save PNG (" << (int)rc << ")" << std::endl;
		return rc;
	}

	rc = am.Save(ad, "blured-global.png");
	if (rc != RCode::Ok) {
		std::cerr << "Failed to save asset (" << (int)rc << ")" << std::endl;
		return rc;
	}
	delete[] ad.data;

	return RC(Ok);	
}


RCode blur_with_shared(AssetsManager& am, byte* source, Png& png) {

	byte* dest;
	CALL_CUDA(cudaMalloc, &dest, png.Height()*png.RowSize());
	cuda_array_raii_t _raii_dest(dest);

	dim3 szblock(cBlockSize, cBlockSize);
	dim3 szgrid(static_cast<int>(std::ceil(png.Width()/(float)cBlockSize)), static_cast<int>(std::ceil(png.Height()/(float)cBlockSize)));
	
	Stopwatch sw;
	sw.start();
	for (u32 i = 0; i < 100; ++i) {
		convolute_in_shared_mem<<<szgrid, szblock>>>(dest, source, png.Width(), png.Height(), png.RowSize(), 3);

		CALL_CUDA(cudaPeekAtLastError);
		CALL_CUDA(cudaDeviceSynchronize);
	}

	float t = sw.measure()/100.0f;
	std::cout << "Shared Elapsed: " << t*1000.0f << "ms" << std::endl;

	CALL_CUDA(cudaMemcpy, png.Data(), dest, png.Height()*png.RowSize(), cudaMemcpyDeviceToHost);

	AssetData ad;
	RCode rc = png.Save(ad);
	if (rc != RCode::Ok) {
		std::cerr << "Failed to save PNG (" << (int)rc << ")" << std::endl;
		return rc;
	}

	rc = am.Save(ad, "blured-shared.png");
	if (rc != RCode::Ok) {
		std::cerr << "Failed to save asset (" << (int)rc << ")" << std::endl;
		return rc;
	}
	delete[] ad.data;

	return RC(Ok);	
}


struct cuda_texture_raii_t {
	using texture_t = texture<byte, cudaTextureType1D, cudaReadModeElementType>;
	texture_t* ptex;

	explicit cuda_texture_raii_t(texture_t* p) : ptex(p) {}
   ~cuda_texture_raii_t() { cudaUnbindTexture(*ptex); }
};


RCode blur_with_texture(AssetsManager& am, byte* source, Png& png) {

	byte* dest;
	CALL_CUDA(cudaMalloc, &dest, png.Height()*png.RowSize());
	volatile cuda_array_raii_t _raii_dest(dest);

	CALL_CUDA(cudaBindTexture, 0, deviceInputTexture, source, png.Height()*png.RowSize());
	volatile cuda_texture_raii_t _raii_tex(&deviceInputTexture);

	dim3 szblock(cBlockSize, cBlockSize);
	dim3 szgrid(static_cast<int>(std::ceil(png.Width()/(float)cBlockSize)), static_cast<int>(std::ceil(png.Height()/(float)cBlockSize)));
	
	Stopwatch sw;
	sw.start();
	for (u32 i = 0; i < 100; ++i) {
		convolute_in_texture<<<szgrid, szblock>>>(dest, source, png.Width(), png.Height(), png.RowSize(), 3);

		CALL_CUDA(cudaPeekAtLastError);
		CALL_CUDA(cudaDeviceSynchronize);
	}

	float t = sw.measure()/100.0f;
	std::cout << "Texture Elapsed: " << t*1000.0f << "ms" << std::endl;

	CALL_CUDA(cudaMemcpy, png.Data(), dest, png.Height()*png.RowSize(), cudaMemcpyDeviceToHost);

	AssetData ad;
	RCode rc = png.Save(ad);
	if (rc != RCode::Ok) {
		std::cerr << "Failed to save PNG (" << (int)rc << ")" << std::endl;
		return rc;
	}

	rc = am.Save(ad, "blured-texture.png");
	if (rc != RCode::Ok) {
		std::cerr << "Failed to save asset (" << (int)rc << ")" << std::endl;
		return rc;
	}
	delete[] ad.data;

	return RC(Ok);	
}


RCode experiment(int argc, char** argv) {

	std::cout << "*********** Lab " << LAB_NUMBER << " ***********" << std::endl;

	if (setup_cuda() != RCode::Ok) {
		return RC(CUDA_Setup);
	}

	int deviceID = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);

	int numSM = deviceProp.multiProcessorCount;
	int maxBlocksPerSM = deviceProp.maxBlocksPerMultiProcessor;
	int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
	int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
	size_t sharedMemPerMultiprocessor = deviceProp.sharedMemPerMultiprocessor;
	size_t sharedMemPerBlock = deviceProp.sharedMemPerBlock;

	std::cout << "SM Number           : " << numSM << std::endl;
	std::cout << "Blocks per SM       : " << maxBlocksPerSM <<std::endl;
	std::cout << "Threads per SM      : " << maxThreadsPerSM <<std::endl;
	std::cout << "Threads per Block   : " << maxThreadsPerBlock << std::endl;
	std::cout << "Shared Mem Per SM   : " << sharedMemPerMultiprocessor / 1024 << "Kb" << std::endl;
	std::cout << "Shared Mem Per Block: " << sharedMemPerBlock / 1024 << "Kb" << std::endl;

	AssetsManager am;
	RCode rc = am.Setup();
	if (rc != RCode::Ok) {
		std::cerr << "Failed to setup Assets Manager (" << (int)rc << ")" << std::endl;
		return rc;
	}

	AssetData ad;
	rc = am.Load(ad, "stonedfox_artsy.png");
	if (rc != RCode::Ok) {
		std::cerr << "Failed to load asset (" << (int)rc << ")" << std::endl;
		return rc;
	}

	Png png;
	rc = png.Load(ad);
	if (rc != RCode::Ok) {
		std::cerr << "Failed to load PNG (" << (int)rc << ")" << std::endl;
		return rc;
	}
	rc = png.Convert(Png::ColorFormat::R8G8B8);
	if (rc != RCode::Ok) {
		std::cerr << "Failed to convert PNG (" << (int)rc << ")" << std::endl;
		return rc;
	}

	CALL_CUDA(cudaMemcpyToSymbol, cDeviceKernel, cKernel, sizeof(cKernel));

	byte* source;
	CALL_CUDA(cudaMalloc, &source, png.Height()*png.RowSize());
	cuda_array_raii_t _raii_source(source);
	CALL_CUDA(cudaMemcpy, source, png.Data(), png.Height()*png.RowSize(), cudaMemcpyHostToDevice);
	
	rc = blur_with_global(am, source, png);
	if (rc != RCode::Ok) return rc;

	rc = blur_with_shared(am, source, png);
	if (rc != RCode::Ok) return rc;

	rc = blur_with_texture(am, source, png);
	if (rc != RCode::Ok) return rc;

	return RC(Ok);
}


int main(int argc, char** argv) {

	return static_cast<int>(experiment(argc, argv));
}

