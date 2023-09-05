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

#ifndef DECODE_NMS_H
#define DECODE_NMS_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#define GPU_BLOCK_THREADS 512

#define checkCudaKernel(...)                                                                                           \
    __VA_ARGS__;                                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = cudaPeekAtLastError();                                                                \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            std::cout << "launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;                             \
        }                                                                                                              \
    } while (0);

void concat_reformat_kernel_invoker(half *input1, half *input2, half *input3, half *output, int C, int height,
                                    int width, cudaStream_t stream);

void decode_nms_validate_kernel_invoker(float *buffers, int num_bboxes, int fm_area, int num_classes,
                                        float confidence_threshold, float nms_threshold, float *affine_matrix,
                                        float *parray, const float *prior_box, int max_objects, cudaStream_t stream);

void decode_nms_validate_kernel_invoker(half *buffers, int num_bboxes, int fm_area, int num_classes,
                                        float confidence_threshold, float nms_threshold, float *affine_matrix,
                                        float *parray, const float *prior_box, int max_objects, cudaStream_t stream);

void decode_nms_kernel_invoker(float *buffers, int num_bboxes, int fm_area, int num_classes, float confidence_threshold,
                               float nms_threshold, float *affine_matrix, float *parray, const float *prior_box,
                               int max_objects, cudaStream_t stream);

void decode_nms_kernel_invoker(half *buffers, int num_bboxes, int fm_area, int num_classes, float confidence_threshold,
                               float nms_threshold, float *affine_matrix, float *parray, const float *prior_box,
                               int max_objects, cudaStream_t stream);

void decode_nms_kernel_invoker(half *buffers, int num_bboxes, int fm_area, int num_classes, float confidence_threshold,
                               float nms_threshold, float *parray, float *prior_box, int max_objects,
                               cudaStream_t stream);

void decode_nms_kernel_invoker(float *buffers, int num_bboxes, int fm_area, int num_classes, float confidence_threshold,
                               float nms_threshold, float *parray, float *prior_box, int max_objects,
                               cudaStream_t stream);

#endif