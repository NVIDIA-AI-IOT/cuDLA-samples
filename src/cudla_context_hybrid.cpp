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

#define DPRINTF(...)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        printf("[hybrid mode] ");                                                                                      \
        printf(__VA_ARGS__);                                                                                           \
    } while (0)

#define CHECK_CUDLA_ERR(m_cudla_err, msg)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (m_cudla_err != cudlaSuccess)                                                                               \
        {                                                                                                              \
            DPRINTF("%s FAILED in %s:%d, CUDLA ERR: %d\n", msg, __FILE__, __LINE__, m_cudla_err);                      \
            exit(1);                                                                                                   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            DPRINTF("%s SUCCESS\n", msg);                                                                              \
        }                                                                                                              \
    } while (0)

#include "cudla_context_hybrid.h"

cuDLAContextHybrid::cuDLAContextHybrid(const char *loadableFilePath)
{
    readDLALoadable(loadableFilePath);
    initialize();
}

void cuDLAContextHybrid::readDLALoadable(const char *loadableFilePath)
{
    FILE *      fp = nullptr;
    struct stat st;
    size_t      actually_read = 0;

    fp = fopen(loadableFilePath, "rb");
    if (fp == nullptr)
    {
        DPRINTF("Cannot open file %s", loadableFilePath);
    }

    if (stat(loadableFilePath, &st) != 0)
    {
        DPRINTF("Cannot open file %s", loadableFilePath);
    }

    m_File_size = st.st_size;

    m_LoadableData = (unsigned char *)malloc(m_File_size);
    if (m_LoadableData == nullptr)
    {
        m_LoadableData = (unsigned char *)malloc(m_File_size);
    }

    actually_read = fread(m_LoadableData, 1, m_File_size, fp);
    if (actually_read != m_File_size)
    {
        free(m_LoadableData);
        DPRINTF("Read wrong size");
    }
    fclose(fp);
}

void cuDLAContextHybrid::initialize()
{
    m_cudla_err = cudlaCreateDevice(0, &m_DevHandle, CUDLA_CUDA_DLA);
    CHECK_CUDLA_ERR(m_cudla_err, "create cuDLA device");

    m_cudla_err = cudlaModuleLoadFromMemory(m_DevHandle, m_LoadableData, m_File_size, &m_ModuleHandle, 0);
    CHECK_CUDLA_ERR(m_cudla_err, "load cuDLA module from memory");

    cudlaModuleAttribute attribute;
    m_cudla_err = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_NUM_INPUT_TENSORS, &attribute);
    CHECK_CUDLA_ERR(m_cudla_err, "cuDLA module get number of input tensors");
    m_NumInputTensors = attribute.numInputTensors;

    m_cudla_err = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_NUM_OUTPUT_TENSORS, &attribute);
    CHECK_CUDLA_ERR(m_cudla_err, "cuDLA module get number of output tensors");
    m_NumOutputTensors = attribute.numOutputTensors;

    m_InputTensorDescs.resize(m_NumInputTensors);
    m_OutputTensorDescs.resize(m_NumOutputTensors);

    attribute.inputTensorDesc = m_InputTensorDescs.data();
    m_cudla_err               = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &attribute);
    CHECK_CUDLA_ERR(m_cudla_err, "cuDLA module get input tensors descriptors");

    attribute.outputTensorDesc = m_OutputTensorDescs.data();
    m_cudla_err                = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &attribute);
    CHECK_CUDLA_ERR(m_cudla_err, "cuDLA module get output tensors descriptors");
}

uint64_t cuDLAContextHybrid::getInputTensorSizeWithIndex(uint8_t index) { return m_InputTensorDescs[index].size; }

uint64_t cuDLAContextHybrid::getOutputTensorSizeWithIndex(uint8_t index) { return m_OutputTensorDescs[index].size; }

uint32_t cuDLAContextHybrid::getNumInputTensors() { return m_NumInputTensors; }

uint32_t cuDLAContextHybrid::getNumOutputTensors() { return m_NumOutputTensors; }

bool cuDLAContextHybrid::initTask(std::vector<void *> &input_dev_buffers, std::vector<void *> &output_dev_buffers)
{
    if (input_dev_buffers.size() != m_NumInputTensors || output_dev_buffers.size() != m_NumOutputTensors)
    {
        DPRINTF("ERROR: number of IO tensors not match\n");
    }

    // Register the CUDA-allocated buffers.
    m_InputBufRegPtrs.resize(m_NumInputTensors);
    m_OutputBufRegPtrs.resize(m_NumOutputTensors);

    for (uint32_t i = 0; i < m_NumInputTensors; i++)
    {
        m_cudla_err = cudlaMemRegister(m_DevHandle, (uint64_t *)input_dev_buffers[i], m_InputTensorDescs[0].size,
                                       &(m_InputBufRegPtrs[i]), 0);
        CHECK_CUDLA_ERR(m_cudla_err, "register cuda input tensor memory to cuDLA");
    }

    for (uint32_t i = 0; i < m_NumOutputTensors; i++)
    {
        m_cudla_err = cudlaMemRegister(m_DevHandle, (uint64_t *)output_dev_buffers[i], m_OutputTensorDescs[i].size,
                                       &(m_OutputBufRegPtrs[i]), 0);
        CHECK_CUDLA_ERR(m_cudla_err, "register cuda output tensor memory to cuDLA");
    }

    m_Task.moduleHandle     = m_ModuleHandle;
    m_Task.inputTensor      = m_InputBufRegPtrs.data();
    m_Task.outputTensor     = m_OutputBufRegPtrs.data();
    m_Task.numInputTensors  = m_NumInputTensors;
    m_Task.numOutputTensors = m_NumOutputTensors;
    m_Task.waitEvents       = nullptr;
    m_Task.signalEvents     = nullptr;

    return true;
}

bool cuDLAContextHybrid::submitDLATask(cudaStream_t stream)
{
    // Enqueue a cuDLA task.
    m_cudla_err = cudlaSubmitTask(m_DevHandle, &m_Task, 1, stream, 0);
    CHECK_CUDLA_ERR(m_cudla_err, "submit cudla task");
    return true;
}

cuDLAContextHybrid::~cuDLAContextHybrid()
{
    for (uint32_t i = 0; i < m_NumInputTensors; i++)
    {
        m_cudla_err = cudlaMemUnregister(m_DevHandle, m_InputBufRegPtrs[i]);
        CHECK_CUDLA_ERR(m_cudla_err, "unregister cudla input ptr");
    }

    for (uint32_t i = 0; i < m_NumOutputTensors; i++)
    {
        m_cudla_err = cudlaMemUnregister(m_DevHandle, m_OutputBufRegPtrs[i]);
        CHECK_CUDLA_ERR(m_cudla_err, "unregister cudla output ptr");
    }

    m_cudla_err    = cudlaModuleUnload(m_ModuleHandle, 0);
    m_ModuleHandle = nullptr;
    CHECK_CUDLA_ERR(m_cudla_err, "unload cudla module");

    m_cudla_err = cudlaDestroyDevice(m_DevHandle);
    m_DevHandle = nullptr;
    CHECK_CUDLA_ERR(m_cudla_err, "unload cudla device handle");

    free(m_LoadableData);
    m_LoadableData = nullptr;
}