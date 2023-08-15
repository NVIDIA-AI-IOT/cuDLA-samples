/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "matx_reformat.h"

#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <chrono>


// #define DEBUG

int test_1()
{
    bool isSuccess = true;
    printf("Test start\n");
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    printf("Create runner\n");
    ReformatRunner* runner = new ReformatRunner();

    // =====================Test input reformat=====================
    const size_t data_size_0 = 1 * 16 * 672 * 672;
    printf("Allocate host buffers\n");
    half* src_host_0 = new half[data_size_0];
    half* dst_host_0 = new half[data_size_0];

    printf("Allocate device buffers\n");
    half* src_dev_0;
    half* dst_dev_0;
    cudaMalloc((void**)&src_dev_0, data_size_0 * sizeof(half));
    cudaMalloc((void**)&dst_dev_0, data_size_0 * sizeof(half));

    std::vector<void*> src_0(1);
    src_0[0] = (char*)src_dev_0;
    std::vector<void*> dst_0(1);
    dst_0[0] = (char*)dst_dev_0;

    half* gt_0 = new half[data_size_0];
    // Initialize input data.
    // 1,3, 672, 672, but pad with 16 channels
    for(int i=0;i<16;i++)
    {
        for(int j=0; j<672*672; j++)
        {
            src_host_0[i*672*672 + j] = __float2half(i+1.f);
        }
    }
    cudaMemcpyAsync(src_dev_0, src_host_0, 1*16*672*672 * sizeof(half), cudaMemcpyHostToDevice, stream);
    runner->ReformatImage(src_0.data(), dst_0.data(), stream);
    cudaMemcpyAsync(dst_host_0, dst_dev_0, data_size_0 * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for(int i=0; i<672*672; i++)
    {
        for(int j=0; j<16; j++)
        {
            gt_0[i*16 + j] = __float2half(j+1.f);
        }
    }
    // Check result;
    for(size_t i=0; i<16*672*672; i++)
    {
        if(dst_host_0[i] != gt_0[i])
        {
            printf("ERROR: Reformat test failed, runner output: %f, ground truth: %f, index: %ld\n",
                __half2float(dst_host_0[i]), __half2float(gt_0[i]), i);
            isSuccess = false;
            break;
        }
    }
    if(isSuccess)
    {
        printf("Test reformat chw -> chw16 PASS\n");
    }
    // =====================Test input reformat=====================

    const size_t data_size_1 = 1 * 256 * (84 * 84 + 42 * 42 + 21 * 21);
    printf("Allocate host buffers\n");
    half* src_host = new half[data_size_1];
    half* dst_host = new half[data_size_1];

    printf("Allocate device buffers\n");
    half* src_dev;
    half* dst_dev;
    cudaMalloc((void**)&src_dev, data_size_1 * sizeof(half));
    cudaMalloc((void**)&dst_dev, data_size_1 * sizeof(half));

    std::vector<void*> src(3);
    src[0] = (char*)src_dev;
    src[1] = (char*)src[0] + 1 * 256 * (84 * 84) * sizeof(half);
    src[2] = (char*)src[1] + 1 * 256 * (42 * 42) * sizeof(half);
    std::vector<void*> dst(3);
    dst[0] = (char*)dst_dev;
    dst[1] = (char*)dst[0] + 1 * 256 * (84 * 84) * sizeof(half);
    dst[2] = (char*)dst[1] + 1 * 256 * (42 * 42) * sizeof(half);

    half* gt = new half[data_size_1];
    // =====================Test reformat=====================
    printf("Test reformat chw16 -> chw\n");

    // Initialize input data.
    // 1,256:16, 84, 84
    for(int i=0;i<16;i++)
    {
        for(int j=0;j<84*84;j++)
        {
            for(int k=0; k<16; k++)
            {
                src_host[i*16*84*84+j*16+k] = __float2half(float(i*16)+float(k)+1);
            }
        }
    }
    // 1,256:16, 41, 42
    for(int i=0;i<16;i++)
    {
        for(int j=0;j<42*42;j++)
        {
            for(int k=0; k<16; k++)
            {
                src_host[1*256*84*84+i*16*42*42+j*16+k] = __float2half(256+float(i*16)+float(k)+1);
            }
        }
    }
    // 1,256:16, 21, 21
    for(int i=0;i<16;i++)
    {
        for(int j=0;j<21*21;j++)
        {
            for(int k=0; k<16; k++)
            {
                src_host[1*256*84*84+1*256*42*42+i*16*21*21+j*16+k] = __float2half(256+256+float(i*16)+float(k)+1);
            }
        }
    }

#ifdef DEBUG
    // Debug
    for(int i=0; i<16; i++)
    {
        for(int j=0;j<16;j++)
        {
            printf("%6.1f ", __half2float(src_host[i*16*84*84+j]));
        }
        printf("\n");
    }
    printf("\n");
#endif
    cudaMemcpyAsync(src_dev, src_host, data_size_1 * sizeof(half), cudaMemcpyHostToDevice, stream);
    runner->Reformat(src.data(), dst.data(), stream);
    cudaMemcpyAsync(dst_host, dst_dev, data_size_1 * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for(int i=0; i<256; i++)
    {
        for(int j=0; j<84*84; j++)
        {
            gt[i*84*84+j] = __float2half(i+1.f);
        }
    }
    for(int i=0; i<256; i++)
    {
        for(int j=0; j<42*42; j++)
        {
            gt[256*84*84+i*42*42+j] = __float2half(256.f+i+1.f);
        }
    }
    for(int i=0; i<256; i++)
    {
        for(int j=0; j<21*21; j++)
        {
            gt[256*84*84+256*42*42+i*21*21+j] = __float2half(512.f+i+1.f);
        }
    }

#ifdef DEBUG
    // Debug
    for(int i=0; i<16; i++)
    {
        for(int j=0;j<16;j++)
        {
            printf("%6.1f ", __half2float(dst_host[i*84*84+j]));
        }
        printf("\n");
    }
    printf("\n");
#endif

    // Check result;
    for(size_t i=0; i<data_size_1; i++)
    {
        if(dst_host[i] != gt[i])
        {
            printf("ERROR: Reformat test failed, runner output: %f, ground truth: %f, index: %ld\n",
                __half2float(dst_host[i]), __half2float(gt[i]), i);
            isSuccess = false;
            break;
        }
    }
    if(isSuccess)
    {
        printf("Test reformat chw16 -> chw PASS\n");
    }
    // =====================Test reformat=====================

    // =====================Test transpose=====================
    printf("Test transpose nhwc -> nchw\n");
    for(int i=0;i<3*85;i++)
    {
        for(int j=0;j<7056;j++)
        {
            src_host[i*7056 + j] = __float2half(j+1.f);
        }
    }
    for(int i=0;i<3*85;i++)
    {
        for(int j=0;j<1764;j++)
        {
            src_host[1*3*85*7056 + i*1764 + j] = __float2half(7056+j+1.f);
        }
    }
    for(int i=0;i<3*85;i++)
    {
        for(int j=0;j<441;j++)
        {
            src_host[1*3*85*7056 + 1*3*85*1764 + i*441 + j] = __float2half(7056+1764+j+1.f);
        }
    }

#ifdef DEBUG
    // Debug
    for(int i=0; i<16; i++)
    {
        for(int j=0;j<16;j++)
        {
            printf("%6.1f ", __half2float(src_host[i*3*85+j]));
        }
        printf("\n");
    }
    printf("\n");
#endif

    src[0] = (char*)src_dev;
    src[1] = (char*)src[0] + 1 * 3 * 85 * 7056 * sizeof(half);
    src[2] = (char*)src[1] + 1 * 3 * 85 * 1764 * sizeof(half);
    dst[0] = (char*)dst_dev;
    dst[1] = (char*)dst[0] + 1 * 3 * 85 * 7056 * sizeof(half);
    dst[2] = (char*)dst[1] + 1 * 3 * 85 * 1764 * sizeof(half);

    cudaMemcpyAsync(src_dev, src_host, 1 * 3 * 85 * 9261 * sizeof(half), cudaMemcpyHostToDevice, stream);
    runner->Transpose(src.data(), dst.data(), stream);
    cudaMemcpyAsync(dst_host, dst_dev, 1 * 3 * 85 * 9261 * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

#ifdef DEBUG
    // Debug
    for(int i=0; i<16; i++)
    {
        for(int j=0;j<16;j++)
        {
            printf("%6.1f ", __half2float(dst_host[i*3*85+j]));
        }
        printf("\n");
    }
    printf("\n");
#endif

    for(int i=0; i<9261; i++)
    {
        for(int j=0; j<3*85; j++)
        {
            gt[i*3*85 + j] = __float2half(i+1);
        }
    }
    // Check result;
    isSuccess = true;
    for(int i=0; i<1*3*85*9261; i++)
    {
        if(dst_host[i] != gt[i])
        {
            printf("ERROR: Transpose test failed, runner output: %f, ground truth: %f, index: %d\n",
                __half2float(dst_host[i]), __half2float(gt[i]), i);
            isSuccess = false;
            break;
        }
    }
    if(isSuccess)
    {
        printf("Test transpose nhwc -> nchw PASS\n");
    }
    // =====================Test transpose=====================

    // =====================Perf test=====================
    printf("Test performance\n");
    half* src_dev_new;
    half* dst_dev_new;
    cudaMalloc((void**)&src_dev_new, 1 * 256 * (84 * 84 + 42 * 42 + 21 * 21) * sizeof(half));
    cudaMalloc((void**)&dst_dev_new, 1 * 9261 * 3 * 85 * sizeof(half));
    std::vector<void*> src_new(3);
    src_new[0] = (char*)src_dev_new;
    src_new[1] = (char*)src_new[0] + 1 * 256 * (84 * 84) * sizeof(half);
    src_new[2] = (char*)src_new[1] + 1 * 256 * (42 * 42) * sizeof(half);
    std::vector<void*> dst_new(3);
    dst_new[0] = (char*)dst_dev_new;
    dst_new[1] = (char*)dst_new[0] + 3 * 85 * 7056 * sizeof(half);
    dst_new[2] = (char*)dst_new[1] + 3 * 85 * 1764 * sizeof(half);
    for(int i=0;i<15;i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        runner->Run(src_new.data(), dst_new.data(), stream);
        cudaStreamSynchronize(stream);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        printf("Execution time: %ld microseconds\n", duration);
    }
    // =====================Perf test=====================

    // Free memory
    cudaStreamDestroy(stream);
    cudaFree(src_dev);
    cudaFree(dst_dev);
    cudaFree(src_dev_new);
    cudaFree(dst_dev_new);
    delete[] src_host;
    delete[] dst_host;
    delete[] gt;
    printf("Test done\n");

    return 0;
}

int test_2()
{
    bool isSuccess = true;
    printf("Test start\n");
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    printf("Create runner\n");
    ReformatRunner* runner = new ReformatRunner();

    // =====================Test input reformat=====================
    printf("Test chw -> hwc4\n");
    const size_t data_size_0 = 1 * 4 * 672 * 672;
    printf("Allocate host buffers\n");
    float* src_host_0 = new float[data_size_0];
    float* dst_host_0 = new float[data_size_0];

    printf("Allocate device buffers\n");
    float* src_dev_0;
    float* dst_dev_0;
    cudaMalloc((void**)&src_dev_0, data_size_0 * sizeof(float));
    cudaMalloc((void**)&dst_dev_0, data_size_0 * sizeof(float));

    std::vector<void*> src_0(1);
    src_0[0] = (char*)src_dev_0;
    std::vector<void*> dst_0(1);
    dst_0[0] = (char*)dst_dev_0;

    float* gt_0 = new float[data_size_0];
    // Initialize input data.
    // 1,3, 672, 672, but pad with 4 channels
    for(int i=0;i<4;i++)
    {
        for(int j=0; j<672*672; j++)
        {
            src_host_0[i*672*672 + j] = i+1.f;
        }
    }
    cudaMemcpyAsync(src_dev_0, src_host_0, 1*4*672*672 * sizeof(float), cudaMemcpyHostToDevice, stream);
    runner->ReformatImageV2(src_0.data(), dst_0.data(), stream);
    cudaMemcpyAsync(dst_host_0, dst_dev_0, data_size_0 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for(int i=0; i<672*672; i++)
    {
        for(int j=0; j<4; j++)
        {
            gt_0[i*4 + j] = j+1.f;
        }
    }
    // Check result;
    for(size_t i=0; i<4*672*672; i++)
    {
        if(dst_host_0[i] != gt_0[i])
        {
            printf("ERROR: Reformat test failed, runner output: %f, ground truth: %f, index: %ld\n",
                __half2float(dst_host_0[i]), __half2float(gt_0[i]), i);
            isSuccess = false;
            break;
        }
    }
    if(isSuccess)
    {
        printf("Test reformat chw -> hwc4 PASS\n");
    }
    // =====================Test input reformat=====================

    const size_t data_size_1 = 1 * 256 * (84 * 84 + 42 * 42 + 21 * 21);
    printf("Allocate host buffers\n");
    half* src_host = new half[data_size_1];
    half* dst_host = new half[data_size_1];

    printf("Allocate device buffers\n");
    half* src_dev;
    half* dst_dev;
    cudaMalloc((void**)&src_dev, data_size_1 * sizeof(half));
    cudaMalloc((void**)&dst_dev, data_size_1 * sizeof(half));

    std::vector<void*> src(3);
    src[0] = (char*)src_dev;
    src[1] = (char*)src[0] + 1 * 256 * (84 * 84) * sizeof(half);
    src[2] = (char*)src[1] + 1 * 256 * (42 * 42) * sizeof(half);
    std::vector<void*> dst(3);
    dst[0] = (char*)dst_dev;
    dst[1] = (char*)dst[0] + 1 * 256 * (84 * 84) * sizeof(half);
    dst[2] = (char*)dst[1] + 1 * 256 * (42 * 42) * sizeof(half);

    half* gt = new half[data_size_1];
    // =====================Test reformat=====================
    printf("Test reformat chw32 -> chw\n");

    // Initialize input data.
    // 1,256:32, 84, 84
    for(int i=0;i<8;i++)
    {
        for(int j=0;j<84*84;j++)
        {
            for(int k=0; k<32; k++)
            {
                src_host[i*32*84*84+j*32+k] = __float2half(float(i*32)+float(k)+1);
            }
        }
    }
    // 1,256:32, 42, 42
    for(int i=0;i<8;i++)
    {
        for(int j=0;j<42*42;j++)
        {
            for(int k=0; k<32; k++)
            {
                src_host[1*256*84*84+i*32*42*42+j*32+k] = __float2half(256+float(i*32)+float(k)+1);
            }
        }
    }
    // 1,256:32, 21, 21
    for(int i=0;i<8;i++)
    {
        for(int j=0;j<21*21;j++)
        {
            for(int k=0; k<32; k++)
            {
                src_host[1*256*84*84+1*256*42*42+i*32*21*21+j*32+k] = __float2half(256+256+float(i*32)+float(k)+1);
            }
        }
    }

#ifdef DEBUG
    // Debug
    for(int i=0; i<8; i++)
    {
        for(int j=0;j<32;j++)
        {
            printf("%6.1f ", __half2float(src_host[i*32*84*84+j]));
        }
        printf("\n");
    }
    printf("\n");
#endif
    cudaMemcpyAsync(src_dev, src_host, data_size_1 * sizeof(half), cudaMemcpyHostToDevice, stream);
    runner->ReformatV2(src.data(), dst.data(), stream);
    cudaMemcpyAsync(dst_host, dst_dev, data_size_1 * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for(int i=0; i<256; i++)
    {
        for(int j=0; j<84*84; j++)
        {
            gt[i*84*84+j] = __float2half(i+1.f);
        }
    }
    for(int i=0; i<256; i++)
    {
        for(int j=0; j<42*42; j++)
        {
            gt[256*84*84+i*42*42+j] = __float2half(256.f+i+1.f);
        }
    }
    for(int i=0; i<256; i++)
    {
        for(int j=0; j<21*21; j++)
        {
            gt[256*84*84+256*42*42+i*21*21+j] = __float2half(512.f+i+1.f);
        }
    }

#ifdef DEBUG
    // Debug
    for(int i=0; i<8; i++)
    {
        for(int j=0;j<32;j++)
        {
            printf("%6.1f ", __half2float(dst_host[i*84*84+j]));
        }
        printf("\n");
    }
    printf("\n");
#endif

    // Check result;
    for(size_t i=0; i<data_size_1; i++)
    {
        if(dst_host[i] != gt[i])
        {
            printf("ERROR: Reformat test failed, runner output: %f, ground truth: %f, index: %ld\n",
                __half2float(dst_host[i]), __half2float(gt[i]), i);
            isSuccess = false;
            break;
        }
    }
    if(isSuccess)
    {
        printf("Test reformat chw32 -> chw PASS\n");
    }
    // =====================Test reformat=====================

    // =====================Perf test=====================
    printf("Test performance\n");
    half* src_dev_new;
    half* dst_dev_new;
    cudaMalloc((void**)&src_dev_new, 1 * 256 * (84 * 84 + 42 * 42 + 21 * 21) * sizeof(half));
    cudaMalloc((void**)&dst_dev_new, 1 * 9261 * 3 * 85 * sizeof(half));
    std::vector<void*> src_new(3);
    src_new[0] = (char*)src_dev_new;
    src_new[1] = (char*)src_new[0] + 1 * 256 * (84 * 84) * sizeof(half);
    src_new[2] = (char*)src_new[1] + 1 * 256 * (42 * 42) * sizeof(half);
    std::vector<void*> dst_new(3);
    dst_new[0] = (char*)dst_dev_new;
    dst_new[1] = (char*)dst_new[0] + 3 * 85 * 7056 * sizeof(half);
    dst_new[2] = (char*)dst_new[1] + 3 * 85 * 1764 * sizeof(half);
    for(int i=0;i<15;i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        runner->RunV2(src_new.data(), dst_new.data(), stream);
        cudaStreamSynchronize(stream);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        printf("Execution time: %ld microseconds\n", duration);
    }
    // =====================Perf test=====================

    // Free memory
    cudaStreamDestroy(stream);
    cudaFree(src_dev);
    cudaFree(dst_dev);
    cudaFree(src_dev_new);
    cudaFree(dst_dev_new);
    delete[] src_host;
    delete[] dst_host;
    delete[] gt;
    printf("Test done\n");

    return 0;
}

int main()
{
    test_1();
    test_2();
}