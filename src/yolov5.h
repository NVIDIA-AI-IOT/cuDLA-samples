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
 
#ifndef __YOLOV5_H__
#define __YOLOV5_H__

#include "NvInfer.h"
#include <NvInferPlugin.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#ifdef USE_DLA_STANDALONE_MODE
#include "cudla_context_standalone.h"
#else
#include "cudla_context_hybrid.h"
#endif
#include "decode_nms.h"
#include "matx_reformat.h"

// opencv for preprocessing &  postprocessing
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define EXIT_SUCCESS 0 /* Successful exit status. */
#define EXIT_FAILURE 1 /* Failing exit status.    */

#define checkCudaErrors(call)                                                                                          \
    {                                                                                                                  \
        cudaError_t ret = (call);                                                                                      \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cout << "Cuda failure: " << cudaGetErrorString(ret) << " at line " << __LINE__ << " in file "         \
                      << __FILE__ << " error status: " << ret << std::endl;                                            \
            abort();                                                                                                   \
        }                                                                                                              \
    }

const int      MAX_IMAGE_BBOX  = 10000;
const int      NUM_BOX_ELEMENT = 7;
const uint32_t NetworkImageWidth{672};
const uint32_t NetworkImageHeight{672};

struct Box
{
    float left, top, right, bottom, confidence;
    float class_label;

    Box() = default;

    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label)
    {
    }
};

typedef std::vector<Box> BoxArray;

enum Yolov5Backend
{
    TRT_GPU    = 1, // NOT USED
    CUDLA_FP16 = 2,
    CUDLA_INT8 = 3,
};

class yolov5
{
  public:
    //!
    //! \brief init yolov5 class object
    //!
    //! \param engine_path The path of engine/loadable file
    //!
    yolov5(std::string engine_path, Yolov5Backend backend);

    //!
    //! \brief release yolov5 class object
    //!
    ~yolov5();

    //!
    //! \brief run tensorRT inference with the data preProcessed
    //!
    int infer();

    // !
    // ! \brief PostProcess, will decode and nms the batch inference result of yolov5
    // !
    // ! \return return all the nms result of yolov5
    // !
    std::vector<std::vector<float>> postProcess(float confidence_threshold, float nms_threshold);

    //!
    //! \brief preprocess a list of image for validate mAP on coco dataset! the model must have a [batchsize, 3, 672,
    //! 672] input
    //!
    //! \param cv_img  input images with BGR-UInt8, the size of the vector must smaller than the max batch size of the
    //! model
    //!
    std::vector<cv::Mat> preProcess4Validate(std::vector<cv::Mat> &cv_img);

    //!
    //! \brief PostProcess for validation on coco dataset, will decode and nms the batch inference result of yolov5 for
    //! mAP test
    //!
    //! \return return all the nms result of yolov5
    //!
    std::vector<std::vector<float>> postProcess4Validation(float confidence_threshold, float nms_threshold);

  private:
    int pushImg(void *imgBuffer, int numImg, bool fromCPU = true);

  private:
    int mImgPushed;
    int mW;
    int mH;

    cudaStream_t mStream;
    float        ms{0.0f};
    cudaEvent_t  mStartEvent, mEndEvent;

    std::vector<void *> mBindingArray;

    std::string   mEnginePath;
    Yolov5Backend mBackend;
#ifdef USE_DLA_STANDALONE_MODE
    cuDLAContextStandalone *mCuDLACtx;
#else
    cuDLAContextHybrid *mCuDLACtx;
#endif

    float               mInputScale   = 0.00787209;
    float               mOutputScale1 = 0.0546086;
    float               mOutputScale2 = 0.148725;
    float               mOutputScale3 = 0.0546086;
    void* mInputTemp;

    // chw16 -> chw -> reshape -> transpose operation for yolov5 heads.
    // also support chw -> chw16 for yolov5 inputs
    ReformatRunner *    mReformatRunner;
    std::vector<void *> mInputCHWPad16;
    std::vector<void *> src;
    std::vector<void *> dst{3};

    // For post-processing
    float *                         mAffineMatrix;
    float *                         prior_ptr_dev;
    float *                         parray_host;
    float *                         parray;
    uint64_t                        parray_size;
    std::vector<std::vector<float>> det_results;
    BoxArray                        bas;
};
#endif