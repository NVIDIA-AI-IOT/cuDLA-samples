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
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cudla.h>

class cuDLAContext
{
  public:
    cuDLAContext(const char *loadableFilePath);
    ~cuDLAContext();

    uint64_t getInputTensorSizeWithIndex(uint8_t index);
    uint64_t getOutputTensorSizeWithIndex(uint8_t index);
    uint32_t getNumInputTensors();
    uint32_t getNumOutputTensors();

    bool initialize();
    bool bufferPrep(void **in_buffers, void **out_buffers, cudaStream_t m_stream);
    bool submitDLATask(cudaStream_t m_stream);
    void cleanUp();

  private:
    bool readDLALoadable(const char *loadableFilePath);
    bool getTensorAttr();

    cudlaDevHandle               m_DevHandle;
    cudlaModule                  m_ModuleHandle;
    unsigned char *              m_LoadableData = NULL;
    uint32_t                     m_NumInputTensors;
    uint32_t                     m_NumOutputTensors;
    cudlaModuleTensorDescriptor *m_InputTensorDesc;
    cudlaModuleTensorDescriptor *m_OutputTensorDesc;
    uint64_t **                  m_InputBufferRegisteredPtr;
    uint64_t **                  m_OutputBufferRegisteredPtr;

    size_t m_File_size;
    bool   m_Initialized;
};

#endif