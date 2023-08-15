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
 
#include "yolov5.h"
#include <cuda_runtime.h>
#include <iostream>

template <typename T, int N> void printBuffer(const void *buffer)
{
    size_t bytes    = N * sizeof(T);
    T *    hostData = new T[N];
    cudaMemcpy(hostData, buffer, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;
    delete[] hostData;
}

yolov5::yolov5(std::string engine_path, Yolov5Backend backend)
{
    mEnginePath = engine_path;
    mBackend    = backend;

    checkCudaErrors(cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking));

    mImgPushed = 0;
    // Malloc cuda memory for binding
    void *input_buf;
    void *output_buf_0;
    void *output_buf_1;
    void *output_buf_2;
    // For FP16 and INT8
    if (mBackend == Yolov5Backend::CUDLA_FP16 || mBackend == Yolov5Backend::CUDLA_INT8)
    {
        // Create a cuda context first, otherwise cudla context creation would fail.
        cudaFree(0);
        mCuDLACtx = new cuDLAContext(engine_path.c_str());

        void *input_buf_before_reformat;
        if (mBackend == Yolov5Backend::CUDLA_FP16)
        {
            // Same size as cuDLA input
            checkCudaErrors(cudaMalloc(&input_buf_before_reformat, mCuDLACtx->getInputTensorSizeWithIndex(0)));
            mInputCHWPad16.push_back(input_buf_before_reformat);
            checkCudaErrors(cudaMalloc(&input_buf, mCuDLACtx->getInputTensorSizeWithIndex(0)));
        }
        if (mBackend == Yolov5Backend::CUDLA_INT8)
        {
            // For int8, we need to do reformat on FP32, then cast to INT8, so we need 4x space.
            checkCudaErrors(
                cudaMalloc(&input_buf_before_reformat, sizeof(float) * mCuDLACtx->getInputTensorSizeWithIndex(0)));
            mInputCHWPad16.push_back(input_buf_before_reformat);
            checkCudaErrors(cudaMalloc(&input_buf, sizeof(float) * mCuDLACtx->getInputTensorSizeWithIndex(0)));
        }
        // Use fp16:chw16 output due to no int8 scale for the output tensor
        checkCudaErrors(cudaMalloc(&output_buf_0, mCuDLACtx->getOutputTensorSizeWithIndex(0)));
        checkCudaErrors(cudaMalloc(&output_buf_1, mCuDLACtx->getOutputTensorSizeWithIndex(1)));
        checkCudaErrors(cudaMalloc(&output_buf_2, mCuDLACtx->getOutputTensorSizeWithIndex(2)));

        std::vector<void *> cudla_inputs{input_buf};
        std::vector<void *> cudla_outputs{output_buf_0, output_buf_1, output_buf_2};
        mCuDLACtx->bufferPrep(cudla_inputs.data(), cudla_outputs.data(), mStream);
    }
    mBindingArray.push_back(reinterpret_cast<void *>(input_buf));
    mBindingArray.push_back(output_buf_0);
    mBindingArray.push_back(output_buf_1);
    mBindingArray.push_back(output_buf_2);

    src = {mBindingArray[1], mBindingArray[2], mBindingArray[3]};
    cudaMalloc((void **)&dst[0], sizeof(half) * 3 * 85 * 9261);
    dst[1] = reinterpret_cast<half*>(dst[0]) + 3 * 85 * 7056;
    dst[2] = reinterpret_cast<half*>(dst[1]) + 3 * 85 * 1764;

    mReformatRunner = new ReformatRunner();

    checkCudaErrors(cudaMalloc(&mAffineMatrix, 3 * sizeof(float)));

    checkCudaErrors(cudaEventCreateWithFlags(&mStartEvent, cudaEventBlockingSync));
    checkCudaErrors(cudaEventCreateWithFlags(&mEndEvent, cudaEventBlockingSync));

    // For post-processing
    float w[16], h[16];
    int   strides[]  = {8, 16, 32};
    int   num_anchor = 3;
    int   num_level  = 3;
    float anchors[]  = {10.000000,  13.000000, 16.000000,  30.000000,  33.000000,  23.000000,
                       30.000000,  61.000000, 62.000000,  45.000000,  59.000000,  119.000000,
                       116.000000, 90.000000, 156.000000, 198.000000, 373.000000, 326.000000};

    int abs_index = 0;
    for (int i = 0; i < num_level; ++i)
    {
        for (int j = 0; j < num_anchor; ++j)
        {
            int aidx = i * num_anchor + j;
            w[aidx]  = anchors[abs_index++];
            h[aidx]  = anchors[abs_index++];
        }
    }

    uint64_t size = 3 * 9261 * 5 * sizeof(float);

    float *prior_ptr = (float *)malloc(size);
    checkCudaErrors(cudaMalloc(&prior_ptr_dev, size));

    float *prior_ptr_start = prior_ptr;
    for (int ianchor = 0; ianchor < num_anchor; ++ianchor)
    {
        for (int ilevel = 0; ilevel < num_level; ++ilevel)
        {
            int stride           = strides[ilevel];
            int fm_width         = NetworkImageWidth / stride;
            int fm_height        = NetworkImageHeight / stride;
            int anchor_abs_index = ilevel * num_anchor + ianchor;
            for (int ih = 0; ih < fm_height; ++ih)
            {
                for (int iw = 0; iw < fm_width; ++iw)
                {
                    *prior_ptr_start++ = iw;
                    *prior_ptr_start++ = ih;
                    *prior_ptr_start++ = w[anchor_abs_index];
                    *prior_ptr_start++ = h[anchor_abs_index];
                    *prior_ptr_start++ = stride;
                }
            }
        }
    }
    checkCudaErrors(cudaMemcpyAsync(prior_ptr_dev, prior_ptr, size, cudaMemcpyHostToDevice, mStream));

    parray_size = (1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT) * sizeof(float);
    checkCudaErrors(cudaMalloc(&parray, parray_size));
    parray_host = (float *)malloc(parray_size);

    return;
}

static void hwc_to_chw(cv::InputArray src, cv::OutputArray dst)
{
    std::vector<cv::Mat> channels;
    cv::split(src, channels);
    // Stretch one-channel images to vector
    for (auto &img : channels)
    {
        img = img.reshape(1, 1);
    }
    // Concatenate three vectors to one
    cv::hconcat(channels, dst);
}

std::vector<cv::Mat> yolov5::preProcess4Validate(std::vector<cv::Mat> &cv_img)
{
    std::vector<cv::Mat> nchwMats;

    for (size_t i = 0; i < cv_img.size(); i++)
    {
        std::vector<float> from_{(float)cv_img[i].cols, (float)cv_img[i].rows};
        mW = cv_img[i].cols;
        mH = cv_img[i].rows;
        std::vector<float> to_{640, 640};
        float              scale = to_[0] / from_[0] < to_[1] / from_[1] ? to_[0] / from_[0] : to_[1] / from_[1];
        std::vector<float> M_{
            scale, 0,     (float)(-scale * from_[0] * 0.5 + to_[0] * 0.5 + scale * 0.5 - 0.5 + 16.0f),
            0,     scale, (float)(-scale * from_[1] * 0.5 + to_[1] * 0.5 + scale * 0.5 - 0.5 + 16.0f)};

        cv::Mat M(2, 3, CV_32FC1, M_.data());
        float   d2i[6];
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i); // dst to image, 2x3 matrix
        cv::invertAffineTransform(M, m2x3_d2i);
        std::vector<float> d2i_1{d2i[0], d2i[2], d2i[5]};

        cv::Mat input_image;
        cv::Mat nchwMat;

        cv::Scalar scalar = cv::Scalar::all(114);
        cv::cvtColor(cv_img[i], input_image, cv::COLOR_BGR2RGB);
        cv::warpAffine(input_image, input_image, M, cv::Size(NetworkImageWidth, NetworkImageHeight), cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, scalar);

        input_image.convertTo(input_image, CV_32FC3, 1.0f / 255.0f, 0);
        hwc_to_chw(input_image, nchwMat);
        nchwMats.push_back(nchwMat);

        this->pushImg(nchwMat.data, 1, true);
        checkCudaErrors(cudaMemcpy(mAffineMatrix, d2i_1.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
    }
    return nchwMats;
}
int yolov5::pushImg(void *imgBuffer, int numImg, bool fromCPU)
{
    if (mBackend == Yolov5Backend::CUDLA_FP16)
    {
        checkCudaErrors(
            cudaMemcpy(mBindingArray[0], imgBuffer, 1 * 3 * 672 * 672 * sizeof(float), cudaMemcpyHostToDevice));
        convert_float_to_half((float *)mBindingArray[0], (__half *)mInputCHWPad16[0], 1 * 3 * 672 * 672);
        mReformatRunner->ReformatImage(mInputCHWPad16.data(), mBindingArray.data(), mStream);
    }
    if (mBackend == Yolov5Backend::CUDLA_INT8)
    {
        checkCudaErrors(
            cudaMemcpy(mBindingArray[0], imgBuffer, 1 * 3 * 672 * 672 * sizeof(float), cudaMemcpyHostToDevice));
        mReformatRunner->ReformatImageV2(mBindingArray.data(), mInputCHWPad16.data(), mStream);
        checkCudaErrors(cudaStreamSynchronize(mStream));
        // dla_hwc4 format, so 1*4*672*672
        convert_float_to_int8((float *)mInputCHWPad16[0], (int8_t *)mBindingArray[0], 1 * 4 * 672 * 672, mInputScale);
    }
    mImgPushed += numImg;
    return 0;
}

int yolov5::infer()
{
    if (mImgPushed == 0)
    {
        std::cerr << " error: mImgPushed = " << mImgPushed << "  ,mImgPushed == 0!" << std::endl;
        return -1;
    }

    checkCudaErrors(cudaEventRecord(mStartEvent, mStream));

    if (mBackend == Yolov5Backend::CUDLA_FP16 || mBackend == Yolov5Backend::CUDLA_INT8)
    {
        mCuDLACtx->submitDLATask(mStream);
    }

    checkCudaErrors(cudaEventRecord(mEndEvent, mStream));
    checkCudaErrors(cudaEventSynchronize(mEndEvent));
    checkCudaErrors(cudaEventElapsedTime(&ms, mStartEvent, mEndEvent));
    std::cout << "Inference time: " << ms << " ms" << std::endl;

    if (mBackend == Yolov5Backend::CUDLA_FP16)
    {
        mReformatRunner->Run(src.data(), dst.data(), mStream);
    }
    if (mBackend == Yolov5Backend::CUDLA_INT8)
    {
        // Use fp16:chw16 output here
        mReformatRunner->Run(src.data(), dst.data(), mStream);
    }

    checkCudaErrors(cudaStreamSynchronize(mStream));
    mImgPushed = 0;
    return 0;
}

std::vector<std::vector<float>> yolov5::postProcess(float confidence_threshold, float nms_threshold)
{
    checkCudaErrors(cudaMemsetAsync(parray, 0, parray_size, mStream));
    memset(parray_host, 0, parray_size);
    decode_nms_kernel_invoker((half *)dst[0],
                              27783, // 9261 * 3,
                              9261, 80, confidence_threshold, nms_threshold, mAffineMatrix, parray, prior_ptr_dev,
                              MAX_IMAGE_BBOX, mStream);
    checkCudaErrors(cudaMemcpyAsync(parray_host, parray, parray_size, cudaMemcpyDeviceToHost, mStream));
    checkCudaErrors(cudaStreamSynchronize(mStream));

    /* Iterate through the inference output for one frame and attach the detected
     * bounding boxes. */
    int count = std::min(MAX_IMAGE_BBOX, (int)*parray_host);
    det_results.clear();
    for (int i = 0; i < count; i++)
    {
        float *pbox = parray_host + 1 + i * NUM_BOX_ELEMENT;

        int keepflag = pbox[6];
        if (keepflag == 0)
            continue;
        float left      = pbox[0];
        float top       = pbox[1];
        float right     = pbox[2];
        float bottom    = pbox[3];
        float confident = pbox[4];
        int   label     = pbox[5];
        det_results.push_back({left, top, right, bottom, confident, (float)label});
    }

    return det_results;
}

// CPU NMS
float limit(float a, float low, float high) { return a < low ? low : (a > high ? high : a); }

static float iou(const Box &a, const Box &b)
{
    float cleft   = std::max(a.left, b.left);
    float ctop    = std::max(a.top, b.top);
    float cright  = std::min(a.right, b.right);
    float cbottom = std::min(a.bottom, b.bottom);

    float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
    float b_area = std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
    return c_area / (a_area + b_area - c_area);
}

static BoxArray cpu_nms(BoxArray &boxes, float threshold)
{

    std::sort(boxes.begin(), boxes.end(),
              [](BoxArray::const_reference a, BoxArray::const_reference b) { return a.confidence > b.confidence; });

    BoxArray output;
    output.reserve(boxes.size());

    std::vector<bool> remove_flags(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
    {

        if (remove_flags[i])
            continue;

        auto &a = boxes[i];
        output.emplace_back(a);

        for (size_t j = i + 1; j < boxes.size(); ++j)
        {
            if (remove_flags[j])
                continue;

            auto &b = boxes[j];
            if (b.class_label == a.class_label)
            {
                if (iou(a, b) >= threshold)
                    remove_flags[j] = true;
            }
        }
    }
    return output;
}

std::vector<std::vector<float>> yolov5::postProcess4Validation(float confidence_threshold, float nms_threshold)
{
    checkCudaErrors(cudaMemsetAsync(parray, 0, parray_size, mStream));
    memset(parray_host, 0, parray_size);
    decode_nms_validate_kernel_invoker((half *)dst[0],
                                       27783, // 9261 * 3,
                                       9261, 80, confidence_threshold, nms_threshold, mAffineMatrix, parray,
                                       prior_ptr_dev, MAX_IMAGE_BBOX, mStream);
    checkCudaErrors(cudaMemcpyAsync(parray_host, parray, parray_size, cudaMemcpyDeviceToHost, mStream));
    checkCudaErrors(cudaStreamSynchronize(mStream));

    /* Iterate through the inference output for one frame and attach the detected
     * bounding boxes. */
    int count = std::min(MAX_IMAGE_BBOX, (int)*parray_host);
    bas.clear();
    for (int i = 0; i < count; i++)
    {
        float *pbox = parray_host + 1 + i * NUM_BOX_ELEMENT;

        int keepflag = pbox[6];
        if (keepflag == 0)
            continue;
        float left      = limit(pbox[0], 0, mW);
        float top       = limit(pbox[1], 0, mH);
        float right     = limit(pbox[2], 0, mW);
        float bottom    = limit(pbox[3], 0, mH);
        float confident = pbox[4];
        float label     = pbox[5];

        bas.emplace_back(left, top, right, bottom, confident, label);
    }
    det_results.clear();
    bas = cpu_nms(bas, nms_threshold);
    for (auto &item : bas)
    {
        det_results.push_back({item.left, item.top, item.right, item.bottom, item.class_label, item.confidence});
    }

    return det_results;
}