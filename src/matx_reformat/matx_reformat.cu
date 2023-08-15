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
#include "matx.h"

#define CHECK_CUDA(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(1); \
    } \
} while(0)

// ----------------------- Half to float -----------------------
__global__ void convert_half_to_float_kernel(const __half* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert half to float and store in b
        b[idx] = __half2float(a[idx]);
    }
}

void convert_half_to_float(__half* a, float* b, int size) {
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_half_to_float_kernel<<<num_blocks, threads_per_block>>>(a, b, size);
}

// ----------------------- Float to half -----------------------
__global__ void convert_float_to_half_kernel(const float* a, __half* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert half to float and store in b
        b[idx] = __float2half(a[idx]);
    }
}

void convert_float_to_half(float* a, __half* b, int size) {
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_float_to_half_kernel<<<num_blocks, threads_per_block>>>(a, b, size);
}

// ----------------------- int8 to half -----------------------
__global__ void convert_int8_to_half_kernel(const int8_t* a, __half* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = static_cast<__half>(a[idx]) * __float2half(scale);
    }
}

void convert_int8_to_half(int8_t* a, __half* b, int size, float scale) {
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_int8_to_half_kernel<<<num_blocks, threads_per_block>>>(a, b, size, scale);
}

// ----------------------- Float to int8 -----------------------
__global__ void convert_float_to_int8_kernel(const float* a, int8_t* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = (a[idx] / scale);
        if(v < -128) v = -128;
        if(v > 127) v = 127;
        b[idx] = (int8_t)v;
    }
}
void convert_float_to_int8(float* a, int8_t* b, int size, float scale){
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_float_to_int8_kernel<<<num_blocks, threads_per_block>>>(a, b, size, scale);
}

void* allocDeviceMemory(size_t size)
{
    void* device_ptr;
    CHECK_CUDA(cudaMalloc(&device_ptr, size));
    return device_ptr;
}

class ReformatRunner::ReformatRunnerImpl {
public:
    ~ReformatRunnerImpl()
    {
        CHECK_CUDA(cudaFree(mTemp));
    }

    void Init()
    {
        mTemp = allocDeviceMemory(sizeof(half)*256*84*84);
    }

    bool Run(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mHead1Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, 16, 84, 84, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead1Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 16, 16, 84, 84});
        matx::tensor_t<matx::matxFp16, 4> mHead1Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 3, 85, 7056});
        matx::tensor_t<matx::matxFp16, 4> mHead1Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, 7056, 3, 85});
        matx::copy(mHead1Output1, mHead1Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead1Output2, mHead1Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead2Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, 16, 42, 42, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead2Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 16, 16, 42, 42});
        matx::tensor_t<matx::matxFp16, 4> mHead2Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 3, 85, 1764});
        matx::tensor_t<matx::matxFp16, 4> mHead2Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, 1764, 3, 85});
        matx::copy(mHead2Output1, mHead2Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead2Output2, mHead2Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead3Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, 16, 21, 21, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead3Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 16, 16, 21, 21});
        matx::tensor_t<matx::matxFp16, 4> mHead3Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 3, 85, 441});
        matx::tensor_t<matx::matxFp16, 4> mHead3Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, 441, 3, 85});
        matx::copy(mHead3Output1, mHead3Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead3Output2, mHead3Input2.Permute({0, 3, 1, 2}), stream);
        return true;
    }

    bool ReformatImage(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mInput1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, 16, 672, 672, 1});
        matx::tensor_t<matx::matxFp16, 5> mOutput1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, 1, 672, 672, 16});
        matx::copy(mOutput1, mInput1.Permute({0, 4, 2, 3, 1}), stream);
        return true;
    }

    bool Reformat(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mHead1Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, 16, 84, 84, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead1Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, 16, 16, 84, 84});
        matx::copy(mHead1Output1, mHead1Input1.Permute({0, 1, 4, 2, 3}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead2Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, 16, 42, 42, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead2Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, 16, 16, 42, 42});
        matx::copy(mHead2Output1, mHead2Input1.Permute({0, 1, 4, 2, 3}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead3Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, 16, 21, 21, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead3Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, 16, 16, 21, 21});
        matx::copy(mHead3Output1, mHead3Input1.Permute({0, 1, 4, 2, 3}), stream);

        return true;
    }

    bool Transpose(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 4> mHead1Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, 3, 85, 7056});
        matx::tensor_t<matx::matxFp16, 4> mHead1Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, 7056, 3, 85});
        matx::copy(mHead1Output2, mHead1Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 4> mHead2Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, 3, 85, 1764});
        matx::tensor_t<matx::matxFp16, 4> mHead2Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, 1764, 3, 85});
        matx::copy(mHead2Output2, mHead2Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 4> mHead3Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, 3, 85, 441});
        matx::tensor_t<matx::matxFp16, 4> mHead3Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, 441, 3, 85});
        matx::copy(mHead3Output2, mHead3Input2.Permute({0, 3, 1, 2}), stream);
        return true;
    }

    bool RunV2(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mHead1Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, 8, 84, 84, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead1Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 8, 32, 84, 84});
        matx::tensor_t<matx::matxFp16, 4> mHead1Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 3, 85, 7056});
        matx::tensor_t<matx::matxFp16, 4> mHead1Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, 7056, 3, 85});
        matx::copy(mHead1Output1, mHead1Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead1Output2, mHead1Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead2Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, 8, 42, 42, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead2Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 8, 32, 42, 42});
        matx::tensor_t<matx::matxFp16, 4> mHead2Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 3, 85, 1764});
        matx::tensor_t<matx::matxFp16, 4> mHead2Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, 1764, 3, 85});
        matx::copy(mHead2Output1, mHead2Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead2Output2, mHead2Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead3Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, 8, 21, 21, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead3Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 8, 32, 21, 21});
        matx::tensor_t<matx::matxFp16, 4> mHead3Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, 3, 85, 441});
        matx::tensor_t<matx::matxFp16, 4> mHead3Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, 441, 3, 85});
        matx::copy(mHead3Output1, mHead3Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead3Output2, mHead3Input2.Permute({0, 3, 1, 2}), stream);
        return true;
    }

    bool ReformatImageV2(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<float, 5> mInput1 = matx::make_tensor<float>((float*)src[0], {1, 4, 672, 672, 1});
        matx::tensor_t<float, 5> mOutput1 = matx::make_tensor<float>((float*)dst[0], {1, 1, 672, 672, 4});
        matx::copy(mOutput1, mInput1.Permute({0, 4, 2, 3, 1}), stream);
        return true;
    }

    bool ReformatV2(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mHead1Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, 8, 84, 84, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead1Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, 8, 32, 84, 84});
        matx::copy(mHead1Output1, mHead1Input1.Permute({0, 1, 4, 2, 3}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead2Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, 8, 42, 42, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead2Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, 8, 32, 42, 42});
        matx::copy(mHead2Output1, mHead2Input1.Permute({0, 1, 4, 2, 3}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead3Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, 8, 21, 21, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead3Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, 8, 32, 21, 21});
        matx::copy(mHead3Output1, mHead3Input1.Permute({0, 1, 4, 2, 3}), stream);

        return true;
    }

private:
    void* mTemp;
};

ReformatRunner::ReformatRunner()
{
    pImpl = new ReformatRunnerImpl();
    pImpl->Init();
}

ReformatRunner::~ReformatRunner()
{
    delete pImpl;
}

bool ReformatRunner::Run(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->Run(src, dst, stream);
}

bool ReformatRunner::ReformatImage(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->ReformatImage(src, dst, stream);
}

bool ReformatRunner::Reformat(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->Reformat(src, dst, stream);
}

bool ReformatRunner::Transpose(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->Transpose(src, dst, stream);
}

bool ReformatRunner::RunV2(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->RunV2(src, dst, stream);
}

bool ReformatRunner::ReformatImageV2(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->ReformatImageV2(src, dst, stream);
}

bool ReformatRunner::ReformatV2(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->ReformatV2(src, dst, stream);
}