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
 
#ifndef MATX_REFORMAT_H
#define MATX_REFORMAT_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// For int8_t
#include <stdint.h>

void convert_half_to_float(__half* a, float* b, int size);

void convert_float_to_half(float* a, __half* b, int size);

void convert_int8_to_half(int8_t* a, __half* b, int size, float scale);

void convert_float_to_int8(float* a, int8_t* b, int size, float scale);

class ReformatRunner
{
public:
    ReformatRunner();
    ~ReformatRunner();

    /**
     * \brief This function perform FP16 chw16 -> chw -> reshape -> transpose operation for yolov5 heads.
     *
     * \param[in] src vector of device chw16 buffer of size 3(1 * 256 * 84 * 84, 1 * 256 * 42 * 42 , 1 * 256 * 21 * 21 )
     * \param[in] dst vector of device chw buffer of size 3(1 * 7056 * 3 * 85, 1 * 1764 * 3 * 85 , 1 * 441 * 3 * 85 )
     */
    bool Run(void** src, void** dst, cudaStream_t stream);

    /**
     * \brief This function perform FP16 nchw->nchw16 conversion for yolov5 input images.
     *
     * \param[in] src vector of device chw buffer of size 1(1*3*672*672, but pad to 16 channels).
     * \param[in] dst vector of device chw16 buffer of size 1 for src.
     */
    bool ReformatImage(void** src, void** dst, cudaStream_t stream);

    // chw16 -> chw reformat, only for test
    bool Reformat(void** src, void** dst, cudaStream_t stream);

    // transpose, only for test
    bool Transpose(void** src, void** dst, cudaStream_t stream);

    /**
     * \brief This function perform FP16 chw32 -> chw -> reshape -> transpose operation for yolov5 heads.
     *
     * \param[in] src vector of device chw32 buffer of size 3(1 * 256 * 84 * 84, 1 * 256 * 42 * 42 , 1 * 256 * 21 * 21 )
     * \param[in] dst vector of device chw buffer of size 3(1 * 7056 * 3 * 85, 1 * 1764 * 3 * 85 , 1 * 441 * 3 * 85 )
     */
    bool RunV2(void** src, void** dst, cudaStream_t stream);

    /**
     * \brief This function perform FP32 nchw->dla_hwc4 conversion for yolov5 input images.
     *
     * \param[in] src vector of device chw buffer of size 1(1*3*672*672, but pad to 4 channels).
     * \param[in] dst vector of device dla_hwc4 buffer of size 1 for src.
     */
    bool ReformatImageV2(void** src, void** dst, cudaStream_t stream);

    // chw32 -> chw reformat, only for test
    bool ReformatV2(void** src, void** dst, cudaStream_t stream);
private:
    class ReformatRunnerImpl;
    ReformatRunnerImpl* pImpl;
};

#endif /* MATX_REFORMAT_H */