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

#include "decode_nms.h"

#define GPU_BLOCK_THREADS 512

dim3 grid_dims(int numJobs)
{
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

dim3 block_dims(int numJobs) { return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS; }

static __host__ inline float desigmoid(float y) { return -log(1.0f / y - 1.0f); }

static __device__ inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static __device__ inline void affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1];
    *oy = matrix[0] * y + matrix[2];
}

static const int NUM_BOX_ELEMENT = 7; // left, top, right, bottom, confidence, class, keepflag

static __global__ void decode_validate_kernel(float *predict, int num_bboxes, int fm_area, int num_classes,
                                              float  confidence_threshold,
                                              float  deconfidence_threshold, // desigmoid
                                              float *affine_matrix, float *parray, const float *prior_box,
                                              int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // prior_box is 25200(axhxw) x 5
    // predict is  9261 x 3 x 85
    int    anchor_index = position / fm_area;
    int    fm_index     = position % fm_area;
    float *pitem        = predict + (fm_index * 3 + anchor_index) * (num_classes + 5);
    float  objectness   = pitem[4];
    if (objectness <= deconfidence_threshold)
        return;

    objectness = sigmoid(objectness);

    float predict_cx = sigmoid(pitem[0]);
    float predict_cy = sigmoid(pitem[1]);
    float predict_w  = sigmoid(pitem[2]);
    float predict_h  = sigmoid(pitem[3]);

    const float *prior_ptr = prior_box + position * 5;
    float        stride    = prior_ptr[4];
    float        cx        = (predict_cx * 2 - 0.5f + prior_ptr[0]) * stride;
    float        cy        = (predict_cy * 2 - 0.5f + prior_ptr[1]) * stride;
    float        width     = pow(predict_w * 2, 2.0f) * prior_ptr[2];
    float        height    = pow(predict_h * 2, 2.0f) * prior_ptr[3];
    float        left      = cx - width * 0.5f;
    float        top       = cy - height * 0.5f;
    float        right     = cx + width * 0.5f;
    float        bottom    = cy + height * 0.5f;
    affine_project(affine_matrix, left, top, &left, &top);
    affine_project(affine_matrix, right, bottom, &right, &bottom);

    for (int i = 0; i < num_classes; ++i)
    {
        float class_confidence = pitem[i + 5];
        if (class_confidence <= deconfidence_threshold)
            continue;

        class_confidence = objectness * sigmoid(class_confidence);
        if (class_confidence <= confidence_threshold)
            continue;

        int index = atomicAdd(parray, 1);
        if (index >= max_objects)
            return;

        float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++     = left;
        *pout_item++     = top;
        *pout_item++     = right;
        *pout_item++     = bottom;
        *pout_item++     = class_confidence;
        *pout_item++     = i;
        *pout_item++     = 1; // 1 = keep, 0 = ignore
    }
}

static __global__ void decode_validate_kernel(half *predict, int num_bboxes, int fm_area, int num_classes,
                                              float  confidence_threshold,
                                              float  deconfidence_threshold, // desigmoid
                                              float *affine_matrix, float *parray, const float *prior_box,
                                              int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // prior_box is 25200(axhxw) x 5
    // predict is  9261 x 3 x 85
    int     anchor_index = position / fm_area;
    int     fm_index     = position % fm_area;
    __half *pitem        = predict + (fm_index * 3 + anchor_index) * (num_classes + 5);
    float   objectness   = __half2float(pitem[4]);
    if (objectness <= deconfidence_threshold)
        return;

    objectness = sigmoid(objectness);

    float predict_cx = sigmoid(__half2float(pitem[0]));
    float predict_cy = sigmoid(__half2float(pitem[1]));
    float predict_w  = sigmoid(__half2float(pitem[2]));
    float predict_h  = sigmoid(__half2float(pitem[3]));

    const float *prior_ptr = prior_box + position * 5;
    float        stride    = prior_ptr[4];
    float        cx        = (predict_cx * 2 - 0.5f + prior_ptr[0]) * stride;
    float        cy        = (predict_cy * 2 - 0.5f + prior_ptr[1]) * stride;
    float        width     = pow(predict_w * 2, 2.0f) * prior_ptr[2];
    float        height    = pow(predict_h * 2, 2.0f) * prior_ptr[3];
    float        left      = cx - width * 0.5f;
    float        top       = cy - height * 0.5f;
    float        right     = cx + width * 0.5f;
    float        bottom    = cy + height * 0.5f;
    affine_project(affine_matrix, left, top, &left, &top);
    affine_project(affine_matrix, right, bottom, &right, &bottom);

    for (int i = 0; i < num_classes; ++i)
    {
        float class_confidence = __half2float(pitem[i + 5]);
        if (class_confidence <= deconfidence_threshold)
            continue;

        class_confidence = objectness * sigmoid(class_confidence);
        if (class_confidence <= confidence_threshold)
            continue;

        int index = atomicAdd(parray, 1);
        if (index >= max_objects)
            return;

        float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
        *pout_item++     = left;
        *pout_item++     = top;
        *pout_item++     = right;
        *pout_item++     = bottom;
        *pout_item++     = class_confidence;
        *pout_item++     = i;
        *pout_item++     = 1; // 1 = keep, 0 = ignore
    }
}

static __global__ void decode_kernel(float *predict, int num_bboxes, int fm_area, int num_classes,
                                     float  confidence_threshold,
                                     float  deconfidence_threshold, // desigmoid
                                     float *affine_matrix, float *parray, const float *prior_box, int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // prior_box is 25200(axhxw) x 5
    // predict is  9261 x 3 x 85
    int    anchor_index = position / fm_area;
    int    fm_index     = position % fm_area;
    float *pitem        = predict + (fm_index * 3 + anchor_index) * (num_classes + 5);
    float  objectness   = pitem[4];
    if (objectness < deconfidence_threshold)
        return;

    float confidence = pitem[5];
    int   label      = 0;
    for (int i = 1; i < num_classes; ++i)
    {
        float class_confidence = pitem[i + 5];
        if (class_confidence > confidence)
        {
            confidence = class_confidence;
            label      = i;
        }
    }

    confidence = sigmoid(confidence);
    objectness = sigmoid(objectness);
    confidence *= objectness;
    if (confidence <= confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float predict_cx = sigmoid(pitem[0]);
    float predict_cy = sigmoid(pitem[1]);
    float predict_w  = sigmoid(pitem[2]);
    float predict_h  = sigmoid(pitem[3]);

    const float *prior_ptr = prior_box + position * 5;
    float        stride    = prior_ptr[4];
    float        cx        = (predict_cx * 2 - 0.5f + prior_ptr[0]) * stride;
    float        cy        = (predict_cy * 2 - 0.5f + prior_ptr[1]) * stride;
    float        width     = pow(predict_w * 2, 2.0f) * prior_ptr[2];
    float        height    = pow(predict_h * 2, 2.0f) * prior_ptr[3];
    float        left      = cx - width * 0.5f;
    float        top       = cy - height * 0.5f;
    float        right     = cx + width * 0.5f;
    float        bottom    = cy + height * 0.5f;
    affine_project(affine_matrix, left, top, &left, &top);
    affine_project(affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++     = left;
    *pout_item++     = top;
    *pout_item++     = right;
    *pout_item++     = bottom;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
}

static __global__ void decode_kernel(half *predict, int num_bboxes, int fm_area, int num_classes,
                                     float  confidence_threshold,
                                     float  deconfidence_threshold, // desigmoid
                                     float *affine_matrix, float *parray, const float *prior_box, int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // prior_box is 25200(axhxw) x 5
    // predict is  9261 x 3 x 85
    int     anchor_index = position / fm_area;
    int     fm_index     = position % fm_area;
    __half *pitem        = predict + (fm_index * 3 + anchor_index) * (num_classes + 5);
    float   objectness   = __half2float(pitem[4]);
    if (objectness < deconfidence_threshold)
        return;

    float confidence = __half2float(pitem[5]);
    int   label      = 0;
    for (int i = 1; i < num_classes; ++i)
    {
        float class_confidence = __half2float(pitem[i + 5]);
        if (class_confidence > confidence)
        {
            confidence = class_confidence;
            label      = i;
        }
    }

    confidence = sigmoid(confidence);
    objectness = sigmoid(objectness);
    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float predict_cx = sigmoid(__half2float(pitem[0]));
    float predict_cy = sigmoid(__half2float(pitem[1]));
    float predict_w  = sigmoid(__half2float(pitem[2]));
    float predict_h  = sigmoid(__half2float(pitem[3]));

    const float *prior_ptr = prior_box + position * 5;
    float        stride    = prior_ptr[4];
    float        cx        = (predict_cx * 2 - 0.5f + prior_ptr[0]) * stride;
    float        cy        = (predict_cy * 2 - 0.5f + prior_ptr[1]) * stride;
    float        width     = pow(predict_w * 2, 2.0f) * prior_ptr[2];
    float        height    = pow(predict_h * 2, 2.0f) * prior_ptr[3];
    float        left      = cx - width * 0.5f;
    float        top       = cy - height * 0.5f;
    float        right     = cx + width * 0.5f;
    float        bottom    = cy + height * 0.5f;
    affine_project(affine_matrix, left, top, &left, &top);
    affine_project(affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++     = left;
    *pout_item++     = top;
    *pout_item++     = right;
    *pout_item++     = bottom;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
}

static __global__ void decode_kernel(__half *predict, int num_bboxes, int fm_area, int num_classes,
                                     float  confidence_threshold,
                                     float  deconfidence_threshold, // desigmoid
                                     float *parray, float *prior_box, int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // prior_box is 25200(axhxw) x 5
    // predict is  9261 x 3 x 85
    int     anchor_index = position / fm_area;
    int     fm_index     = position % fm_area;
    __half *pitem        = predict + (fm_index * 3 + anchor_index) * (num_classes + 5);
    float   objectness   = __half2float(pitem[4]);
    if (objectness < deconfidence_threshold)
        return;

    float confidence = __half2float(pitem[5]);
    int   label      = 0;
    for (int i = 1; i < num_classes; ++i)
    {
        float class_confidence = __half2float(pitem[i + 5]);
        if (class_confidence > confidence)
        {
            confidence = class_confidence;
            label      = i;
        }
    }

    confidence = sigmoid(confidence);
    objectness = sigmoid(objectness);
    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float predict_cx = sigmoid(__half2float(pitem[0]));
    float predict_cy = sigmoid(__half2float(pitem[1]));
    float predict_w  = sigmoid(__half2float(pitem[2]));
    float predict_h  = sigmoid(__half2float(pitem[3]));

    const float *prior_ptr = prior_box + position * 5;
    float        stride    = prior_ptr[4];
    float        cx        = (predict_cx * 2 - 0.5f + prior_ptr[0]) * stride;
    float        cy        = (predict_cy * 2 - 0.5f + prior_ptr[1]) * stride;
    float        width     = pow(predict_w * 2, 2.0f) * prior_ptr[2];
    float        height    = pow(predict_h * 2, 2.0f) * prior_ptr[3];
    float        left      = cx - width * 0.5f;
    float        top       = cy - height * 0.5f;
    float        right     = cx + width * 0.5f;
    float        bottom    = cy + height * 0.5f;

    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++     = left;
    *pout_item++     = top;
    *pout_item++     = right;
    *pout_item++     = bottom;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
}

static __global__ void decode_kernel(float *predict, int num_bboxes, int fm_area, int num_classes,
                                     float  confidence_threshold,
                                     float  deconfidence_threshold, // desigmoid
                                     float *parray, float *prior_box, int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // prior_box is 25200(axhxw) x 5
    // predict is  9261 x 3 x 85
    int    anchor_index = position / fm_area;
    int    fm_index     = position % fm_area;
    float *pitem        = predict + (fm_index * 3 + anchor_index) * (num_classes + 5);
    float  objectness   = pitem[4];
    if (objectness < deconfidence_threshold)
        return;

    float confidence = pitem[5];
    int   label      = 0;
    for (int i = 1; i < num_classes; ++i)
    {
        float class_confidence = pitem[i + 5];
        if (class_confidence > confidence)
        {
            confidence = class_confidence;
            label      = i;
        }
    }

    confidence = sigmoid(confidence);
    objectness = sigmoid(objectness);
    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float predict_cx = sigmoid(pitem[0]);
    float predict_cy = sigmoid(pitem[1]);
    float predict_w  = sigmoid(pitem[2]);
    float predict_h  = sigmoid(pitem[3]);

    const float *prior_ptr = prior_box + position * 5;
    float        stride    = prior_ptr[4];
    float        cx        = (predict_cx * 2 - 0.5f + prior_ptr[0]) * stride;
    float        cy        = (predict_cy * 2 - 0.5f + prior_ptr[1]) * stride;
    float        width     = pow(predict_w * 2, 2.0f) * prior_ptr[2];
    float        height    = pow(predict_h * 2, 2.0f) * prior_ptr[3];
    float        left      = cx - width * 0.5f;
    float        top       = cy - height * 0.5f;
    float        right     = cx + width * 0.5f;
    float        bottom    = cy + height * 0.5f;

    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++     = left;
    *pout_item++     = top;
    *pout_item++     = right;
    *pout_item++     = bottom;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop,
                                float bright, float bbottom)
{

    float cleft   = max(aleft, bleft);
    float ctop    = max(atop, btop);
    float cright  = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void nms_kernel(float *bboxes, int max_objects, float threshold)
{

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count    = min((int)*bboxes, max_objects);
    if (position >= count)
        return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i)
    {
        float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5])
            continue;

        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou =
                box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou >= threshold)
            {
                pcurrent[6] = 0; // 1=keep, 0=ignore
                return;
            }
        }
    }
}

void decode_nms_validate_kernel_invoker(float *predict, int num_bboxes, int fm_area, int num_classes,
                                        float confidence_threshold, float nms_threshold, float *affine_matrix,
                                        float *parray, const float *prior_box, int max_objects, cudaStream_t stream)
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkCudaKernel(decode_validate_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, fm_area, num_classes, confidence_threshold, desigmoid(confidence_threshold), affine_matrix,
        parray, prior_box, max_objects));
}

void decode_nms_validate_kernel_invoker(half *predict, int num_bboxes, int fm_area, int num_classes,
                                        float confidence_threshold, float nms_threshold, float *affine_matrix,
                                        float *parray, const float *prior_box, int max_objects, cudaStream_t stream)
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkCudaKernel(decode_validate_kernel<<<grid, block, 0, stream>>>(
        (half *)predict, num_bboxes, fm_area, num_classes, confidence_threshold, desigmoid(confidence_threshold),
        affine_matrix, parray, prior_box, max_objects));
}

void decode_nms_kernel_invoker(float *predict, int num_bboxes, int fm_area, int num_classes, float confidence_threshold,
                               float nms_threshold, float *affine_matrix, float *parray, const float *prior_box,
                               int max_objects, cudaStream_t stream)
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, fm_area, num_classes,
                                                              confidence_threshold, desigmoid(confidence_threshold),
                                                              affine_matrix, parray, prior_box, max_objects));
}

void decode_nms_kernel_invoker(half *predict, int num_bboxes, int fm_area, int num_classes, float confidence_threshold,
                               float nms_threshold, float *affine_matrix, float *parray, const float *prior_box,
                               int max_objects, cudaStream_t stream)
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>((half *)predict, num_bboxes, fm_area, num_classes,
                                                              confidence_threshold, desigmoid(confidence_threshold),
                                                              affine_matrix, parray, prior_box, max_objects));
}

void decode_nms_kernel_invoker(half *predict, int num_bboxes, int fm_area, int num_classes, float confidence_threshold,
                               float nms_threshold, float *parray, float *prior_box, int max_objects,
                               cudaStream_t stream)
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, fm_area, num_classes,
                                                              confidence_threshold, desigmoid(confidence_threshold),
                                                              parray, prior_box, max_objects));

    grid  = grid_dims(max_objects);
    block = block_dims(max_objects);
    checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
}

void decode_nms_kernel_invoker(float *predict, int num_bboxes, int fm_area, int num_classes, float confidence_threshold,
                               float nms_threshold, float *parray, float *prior_box, int max_objects,
                               cudaStream_t stream)
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkCudaKernel(decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, fm_area, num_classes,
                                                              confidence_threshold, desigmoid(confidence_threshold),
                                                              parray, prior_box, max_objects));

    grid  = grid_dims(max_objects);
    block = block_dims(max_objects);
    checkCudaKernel(nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
}