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

#ifndef CUDLA_CONTEXT_STANDALONE_H
#define CUDLA_CONTEXT_STANDALONE_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "cudla.h"
#include "nvscibuf.h"
#include "nvscierror.h"
#include "nvscisync.h"

#include <vector>

struct NvSciBufferContext
{
    // cudla_tensor_desc is a pointer to the element of
    // m_Input_Tensor_Descs or m_Output_Tensor_Descs,
    // so explicitly release is unecessary.
    cudlaModuleTensorDescriptor * cudla_tensor_desc;
    NvSciBufAttrList              unreconciled_attr_list;
    NvSciBufAttrList              reconciled_attr_list;
    NvSciBufAttrList              conflict_list;
    NvSciBufObj                   buf_obj;
    cudlaExternalMemoryHandleDesc cudla_ext_mem_desc;
    uint64_t *                    buf_registered_ptr_cudla;
    cudaExternalMemoryHandleDesc  cuda_mem_handle_desc;
    cudaExternalMemoryBufferDesc  cuda_ext_buffer_desc;
    cudaExternalMemory_t          ext_mem_raw_buf;
    void *                        buf_gpu;
};

struct NvSciSyncContext
{
    NvSciSyncAttrList                signaler_attr_list;
    NvSciSyncAttrList                waiter_attr_list;
    NvSciSyncAttrList                reconciled_attr_list;
    NvSciSyncAttrList                conflict_list;
    NvSciSyncObj                     sync_obj;
    uint64_t *                       nvsci_sync_obj_reg_ptr;
    cudlaExternalSemaphoreHandleDesc cudla_ext_sema_mem_desc;
    NvSciSyncFence *                 nvsci_fence_ptr;
    CudlaFence *                     cudla_fence_ptr;
};

class cuDLAContextStandalone
{
  public:
    //!
    //! \brief Construct infer context from loadable
    //!
    cuDLAContextStandalone(const char *loadableFilePath);

    //!
    //! \brief Deconstructor
    //!
    ~cuDLAContextStandalone();

    //!
    //! \brief Get input tensor size of index
    //!
    uint64_t getInputTensorSizeWithIndex(int32_t index);

    //!
    //! \brief Get output tensor size of index
    //!
    uint64_t getOutputTensorSizeWithIndex(int32_t index);

    //!
    //! \brief Get number of input tensors
    //!
    uint32_t getNumInputTensors();

    //!
    //! \brief Get number of output tensors
    //!
    uint32_t getNumOutputTensors();

    //!
    //! \brief Get input CUDA buffer ptr
    //!
    void *getInputCudaBufferPtr(int32_t index);

    //!
    //! \brief Get output CUDA buffer ptr
    //!
    void *getOutputCudaBufferPtr(int32_t index);

    //!
    //! \brief launch cuDLA task
    //!
    int submitDLATask(cudaStream_t streamToRun);

  private:
    bool readDLALoadable(const char *loadableFilePath);
    bool initialize();
    void releaseNvSciBufferContexts(std::vector<NvSciBufferContext> &contexts);
    void releaseNvSciSyncContext(NvSciSyncContext &context);

    bool        m_Has_initialized;
    cudlaStatus m_cudla_err;
    cudaError   m_cuda_err;
    NvSciError  m_nvsci_err;

    unsigned char *m_LoadableData = nullptr;
    size_t         m_File_size;

    uint32_t                        m_NumInputTensors  = 0;
    uint32_t                        m_NumOutputTensors = 0;
    std::vector<NvSciBufferContext> m_InputsBufContext;
    std::vector<NvSciBufferContext> m_OutputsBufContext;
    // m_InputBufRegPtrs and m_OutputBufRegPtrs are
    // associate with std::vector<NvSciBufferContext>
    // so we don't need to release it explicitly.
    std::vector<uint64_t *>                  m_InputBufRegPtrs;
    std::vector<uint64_t *>                  m_OutputBufRegPtrs;
    std::vector<cudlaModuleTensorDescriptor> m_Input_Tensor_Descs;
    std::vector<cudlaModuleTensorDescriptor> m_Output_Tensor_Descs;

    cudlaDevHandle       m_DevHandle;
    cudlaModule          m_ModuleHandle;
    cudlaModuleAttribute m_cudla_module_attribute;

    NvSciBufModule  m_NvSciBufModule;
    NvSciSyncModule m_NvSciSyncModule;

    cudlaWaitEvents *               m_WaitEvents;
    NvSciSyncContext                m_WaitEventContext;
    cudaExternalSemaphore_t         m_WaitSem;
    cudaExternalSemaphoreWaitParams m_WaitParams;

    cudlaSignalEvents *               m_SignalEvents;
    NvSciSyncContext                  m_SignalEventContext;
    cudaExternalSemaphore_t           m_SignalSem;
    cudaExternalSemaphoreSignalParams m_SignalParams;

    cudlaTask m_Task;
};

#endif
