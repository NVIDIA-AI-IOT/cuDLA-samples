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

#ifndef CUDLA_CONTEXT_H
#define CUDLA_CONTEXT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cudla.h>

class cuDLAContextHybrid
{
  public:
    cuDLAContextHybrid(const char *loadableFilePath);
    ~cuDLAContextHybrid();

    uint64_t getInputTensorSizeWithIndex(uint8_t index);
    uint64_t getOutputTensorSizeWithIndex(uint8_t index);
    uint32_t getNumInputTensors();
    uint32_t getNumOutputTensors();

    bool initTask(std::vector<void *> &input_dev_buffers, std::vector<void *> &output_dev_buffers);
    bool submitDLATask(cudaStream_t stream);

  private:
    bool                                     initialize();
    bool                                     readDLALoadable(const char *loadableFilePath);
    cudlaStatus                              m_cudla_err;
    cudlaDevHandle                           m_DevHandle;
    cudlaModule                              m_ModuleHandle;
    unsigned char *                          m_LoadableData = nullptr;
    size_t                                   m_File_size;
    uint32_t                                 m_NumInputTensors;
    uint32_t                                 m_NumOutputTensors;
    std::vector<cudlaModuleTensorDescriptor> m_InputTensorDescs;
    std::vector<cudlaModuleTensorDescriptor> m_OutputTensorDescs;
    std::vector<uint64_t *>                  m_InputBufRegPtrs;
    std::vector<uint64_t *>                  m_OutputBufRegPtrs;

    cudlaTask m_Task;
};

#endif