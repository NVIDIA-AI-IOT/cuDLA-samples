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
 
#include "cudla_context.h"

cuDLAContext::cuDLAContext(const char *loadableFilePath)
{
    readDLALoadable(loadableFilePath);
    m_Initialized = false;
    if (!m_Initialized)
    {
        initialize();
    }
    else
    {
        std::cerr << "The cuDLAContext was initialized." << std::endl;
    }
}

bool cuDLAContext::readDLALoadable(const char *loadableFilePath)
{
    FILE *      fp = NULL;
    struct stat st;
    size_t      actually_read = 0;

    // Read loadable into buffer.
    fp = fopen(loadableFilePath, "rb");
    if (fp == NULL)
    {
        std::cerr << "Cannot open file " << loadableFilePath << std::endl;
        return false;
    }

    if (stat(loadableFilePath, &st) != 0)
    {
        std::cerr << "Cannot stat file" << std::endl;
        return false;
    }

    m_File_size = st.st_size;

    m_LoadableData = (unsigned char *)malloc(m_File_size);
    if (m_LoadableData == NULL)
    {
        std::cerr << "Cannot Allocate memory for loadable" << std::endl;
        return false;
    }

    actually_read = fread(m_LoadableData, 1, m_File_size, fp);
    if (actually_read != m_File_size)
    {
        free(m_LoadableData);
        std::cerr << "Read wrong size" << std::endl;
        return false;
    }
    fclose(fp);
    return true;
}

bool cuDLAContext::initialize()
{
    cudlaStatus err;
    err = cudlaCreateDevice(0, &m_DevHandle, CUDLA_CUDA_DLA);
    if (err != cudlaSuccess)
    {
        std::cerr << "Error in cuDLA create device = " << err << std::endl;
        cleanUp();
        return false;
    }

    err = cudlaModuleLoadFromMemory(m_DevHandle, m_LoadableData, m_File_size, &m_ModuleHandle, 0);
    if (err != cudlaSuccess)
    {
        std::cerr << "Error in cudlaModuleLoadFromMemory = " << err << std::endl;
        cleanUp();
        return false;
    }

    m_Initialized = true;
    if (!getTensorAttr())
    {
        cleanUp();
        return false;
    }
    std::cout << "DLA CTX INIT !!!" << std::endl;
    return true;
}

uint64_t cuDLAContext::getInputTensorSizeWithIndex(uint8_t index) { return m_InputTensorDesc[index].size; }

uint64_t cuDLAContext::getOutputTensorSizeWithIndex(uint8_t index) { return m_OutputTensorDesc[index].size; }

uint32_t cuDLAContext::getNumInputTensors() { return m_NumInputTensors; }

uint32_t cuDLAContext::getNumOutputTensors() { return m_NumOutputTensors; }

bool cuDLAContext::getTensorAttr()
{
    if (!m_Initialized)
    {
        return false;
    }
    // Get tensor attributes.
    cudlaStatus          err;
    cudlaModuleAttribute attribute;

    err = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_NUM_INPUT_TENSORS, &attribute);
    if (err != cudlaSuccess)
    {
        std::cerr << "Error in getting numInputTensors = " << err << std::endl;
        cleanUp();
        return false;
    }
    m_NumInputTensors = attribute.numInputTensors;

    err = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_NUM_OUTPUT_TENSORS, &attribute);
    if (err != cudlaSuccess)
    {
        std::cerr << "Error in getting numOutputTensors = " << err << std::endl;
        cleanUp();
        return false;
    }
    m_NumOutputTensors = attribute.numOutputTensors;

    m_InputTensorDesc = (cudlaModuleTensorDescriptor *)malloc(sizeof(cudlaModuleTensorDescriptor) * m_NumInputTensors);
    m_OutputTensorDesc =
        (cudlaModuleTensorDescriptor *)malloc(sizeof(cudlaModuleTensorDescriptor) * m_NumOutputTensors);
    if ((m_InputTensorDesc == NULL) || (m_OutputTensorDesc == NULL))
    {
        std::cerr << "Error in allocating memory for TensorDesc" << std::endl;
        cleanUp();
        return false;
    }

    attribute.inputTensorDesc = m_InputTensorDesc;
    err                       = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &attribute);
    if (err != cudlaSuccess)
    {
        std::cerr << "Error in getting input tensor descriptor = " << err << std::endl;
        cleanUp();
        return false;
    }

    attribute.outputTensorDesc = m_OutputTensorDesc;
    err                        = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &attribute);
    if (err != cudlaSuccess)
    {
        std::cerr << "Error in getting output tensor descriptor = " << err << std::endl;
        cleanUp();
        return false;
    }

    return true;
}

bool cuDLAContext::bufferPrep(void **in_buffers, void **out_buffers, cudaStream_t m_stream)
{
    if (!m_Initialized)
    {
        return false;
    }
    cudlaStatus err;

    // Register the CUDA-allocated buffers.
    m_InputBufferRegisteredPtr  = (uint64_t **)malloc(sizeof(uint64_t *) * m_NumInputTensors);
    m_OutputBufferRegisteredPtr = (uint64_t **)malloc(sizeof(uint64_t *) * m_NumOutputTensors);

    if ((m_InputBufferRegisteredPtr == NULL) || (m_OutputBufferRegisteredPtr == NULL))
    {
        std::cerr << "Error in allocating memory for BufferRegisteredPtr" << std::endl;
        cleanUp();
        return false;
    }

    for (uint32_t ii = 0; ii < m_NumInputTensors; ii++)
    {
        err = cudlaMemRegister(m_DevHandle, (uint64_t *)(in_buffers[ii]), m_InputTensorDesc[0].size,
                               &(m_InputBufferRegisteredPtr[0]), 0);
        if (err != cudlaSuccess)
        {
            std::cerr << "Error in registering input tensor memory " << ii << ": " << err << std::endl;
            cleanUp();
            return false;
        }
    }

    for (uint32_t ii = 0; ii < m_NumOutputTensors; ii++)
    {
        err = cudlaMemRegister(m_DevHandle, (uint64_t *)(out_buffers[ii]), m_OutputTensorDesc[ii].size,
                               &(m_OutputBufferRegisteredPtr[ii]), 0);
        if (err != cudlaSuccess)
        {
            std::cerr << "Error in registering output tensor memory " << ii << ": " << err << std::endl;
            cleanUp();
            return false;
        }
    }

    std::cout << "ALL MEMORY REGISTERED SUCCESSFULLY" << std::endl;

    // Memset output buffer on GPU to 0.
    for (uint32_t ii = 0; ii < m_NumOutputTensors; ii++)
    {
        cudaError_t result = cudaMemsetAsync(out_buffers[ii], 0, m_OutputTensorDesc[ii].size, m_stream);
        if (result != cudaSuccess)
        {
            std::cerr << "Error in enqueueing memset for output tensor " << ii << std::endl;
            cleanUp();
            return false;
        }
    }

    return true;
}

bool cuDLAContext::submitDLATask(cudaStream_t m_stream)
{
    std::cout << "SUBMIT CUDLA TASK" << std::endl;
    std::cout << "    Input Tensor Num: " << m_NumInputTensors << std::endl;
    std::cout << "    Output Tensor Num: " << m_NumOutputTensors << std::endl;

    cudlaTask task;
    task.moduleHandle     = m_ModuleHandle;
    task.inputTensor      = m_InputBufferRegisteredPtr;
    task.outputTensor     = m_OutputBufferRegisteredPtr;
    task.numOutputTensors = m_NumOutputTensors;
    task.numInputTensors  = m_NumInputTensors;
    task.waitEvents       = NULL;
    task.signalEvents     = NULL;

    // Enqueue a cuDLA task.
    cudlaStatus err = cudlaSubmitTask(m_DevHandle, &task, 1, m_stream, 0);
    if (err != cudlaSuccess)
    {
        std::cerr << "Error in submitting task : " << err << std::endl;
        cleanUp();
        return false;
    }

    std::cout << "SUBMIT IS DONE !!!" << std::endl;

    return true;
}

void cuDLAContext::cleanUp()
{
    if (m_InputTensorDesc != NULL)
    {
        free(m_InputTensorDesc);
        m_InputTensorDesc = NULL;
    }
    if (m_OutputTensorDesc != NULL)
    {
        free(m_OutputTensorDesc);
        m_OutputTensorDesc = NULL;
    }

    if (m_LoadableData != NULL)
    {
        free(m_LoadableData);
        m_LoadableData = NULL;
    }

    if (m_ModuleHandle != NULL)
    {
        cudlaModuleUnload(m_ModuleHandle, 0);
        m_ModuleHandle = NULL;
    }

    if (m_DevHandle != NULL)
    {
        cudlaDestroyDevice(m_DevHandle);
        m_DevHandle = NULL;
    }

    if (m_InputBufferRegisteredPtr != NULL)
    {
        free(m_InputBufferRegisteredPtr);
        m_InputBufferRegisteredPtr = NULL;
    }

    if (m_OutputBufferRegisteredPtr != NULL)
    {
        free(m_OutputBufferRegisteredPtr);
        m_OutputBufferRegisteredPtr = NULL;
    }

    m_NumInputTensors  = 0;
    m_NumOutputTensors = 0;
    std::cout << "DLA CTX CLEAN UP !!!" << std::endl;
}

cuDLAContext::~cuDLAContext()
{
    cudlaStatus err;
    for (uint32_t ii = 0; ii < m_NumInputTensors; ii++)
    {
        err = cudlaMemUnregister(m_DevHandle, m_InputBufferRegisteredPtr[ii]);
        if (err != cudlaSuccess)
        {
            std::cerr << "Error in unregistering input tensor memory " << ii << ": " << err << std::endl;
        }
    }

    for (uint32_t ii = 0; ii < m_NumOutputTensors; ii++)
    {
        err = cudlaMemUnregister(m_DevHandle, m_OutputBufferRegisteredPtr[ii]);
        if (err != cudlaSuccess)
        {
            std::cerr << "Error in unregistering output tensor memory " << ii << ": " << err << std::endl;
        }
    }
}