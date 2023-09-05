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
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#include "cudla_context_standalone.h"

#define DPRINTF(...)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        printf("[standalone mode] ");                                                                                  \
        printf(__VA_ARGS__);                                                                                           \
    } while (0)

#define CHECK_CUDA_ERR(err, msg)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            DPRINTF("%s FAILED in %s:%d, CUDA ERR: %d\n", msg, __FILE__, __LINE__, err);                               \
            exit(1);                                                                                                   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            DPRINTF("%s SUCCESS\n", msg);                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDLA_ERR(err, msg)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if (err != cudlaSuccess)                                                                                       \
        {                                                                                                              \
            DPRINTF("%s FAILED in %s:%d, CUDLA ERR: %d\n", msg, __FILE__, __LINE__, err);                              \
            exit(1);                                                                                                   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            DPRINTF("%s SUCCESS\n", msg);                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_NVSCI_ERR(err, msg)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if (err != NvSciError_Success)                                                                                 \
        {                                                                                                              \
            DPRINTF("%s FAILED in %s:%d, NVSCI ERR: %d\n", msg, __FILE__, __LINE__, err);                              \
            exit(1);                                                                                                   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            DPRINTF("%s SUCCESS\n", msg);                                                                              \
        }                                                                                                              \
    } while (0)

static void printTensorDesc(cudlaModuleTensorDescriptor *tensorDesc)
{
    DPRINTF("TENSOR NAME : %s\n", tensorDesc->name);
    DPRINTF("size: %lu\n", tensorDesc->size);
    DPRINTF("dims: [%lu, %lu, %lu, %lu]\n", tensorDesc->n, tensorDesc->c, tensorDesc->h, tensorDesc->w);
    DPRINTF("data fmt: %d\n", tensorDesc->dataFormat);
    DPRINTF("data type: %d\n", tensorDesc->dataType);
    DPRINTF("data category: %d\n", tensorDesc->dataCategory);
    DPRINTF("pixel fmt: %d\n", tensorDesc->pixelFormat);
    DPRINTF("pixel mapping: %d\n", tensorDesc->pixelMapping);
    DPRINTF("stride[0]: %d\n", tensorDesc->stride[0]);
    DPRINTF("stride[1]: %d\n", tensorDesc->stride[1]);
    DPRINTF("stride[2]: %d\n", tensorDesc->stride[2]);
    DPRINTF("stride[3]: %d\n", tensorDesc->stride[3]);
}

void createAndSetAttrList(NvSciBufModule module, uint64_t bufSize, NvSciBufAttrList *attrList)
{
    NvSciError sciStatus;
    sciStatus = NvSciBufAttrListCreate(module, attrList);
    CHECK_NVSCI_ERR(sciStatus, "create NvSci buffer attr list");

    NvSciBufType              bufType         = NvSciBufType_RawBuffer;
    bool                      cpu_access_flag = false;
    NvSciBufAttrValAccessPerm perm            = NvSciBufAccessPerm_ReadWrite;

    CUuuid   uuid;
    CUresult res = cuDeviceGetUuid(&uuid, 0);
    if (res != CUDA_SUCCESS)
    {
        fprintf(stderr, "Driver API error = %04d \n", res);
    }

    // Fill in values
    NvSciBufAttrKeyValuePair raw_buf_attrs[] = {
        {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
        {NvSciBufRawBufferAttrKey_Size, &bufSize, sizeof(bufSize)},
        {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpu_access_flag, sizeof(cpu_access_flag)},
        {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},
        {NvSciBufGeneralAttrKey_GpuId, &uuid, sizeof(uuid)},
    };
    size_t length = sizeof(raw_buf_attrs) / sizeof(NvSciBufAttrKeyValuePair);

    sciStatus = NvSciBufAttrListSetAttrs(*attrList, raw_buf_attrs, length);
    CHECK_NVSCI_ERR(sciStatus, "set NvSci buffer attr list");
}

cuDLAContextStandalone::cuDLAContextStandalone(const char *loadableFilePath)
{
    readDLALoadable(loadableFilePath);
    initialize();
    m_Has_initialized = true;
}

// Read loadable in to memory
void cuDLAContextStandalone::readDLALoadable(const char *loadableFilePath)
{
    FILE *      fp = nullptr;
    struct stat st;
    size_t      actually_read = 0;

    fp = fopen(loadableFilePath, "rb");
    if (fp == nullptr)
    {
        DPRINTF("Cannot open file %s", loadableFilePath);
        exit(1);
    }

    if (stat(loadableFilePath, &st) != 0)
    {
        DPRINTF("Cannot open file %s", loadableFilePath);
        exit(1);
    }

    m_File_size = st.st_size;

    m_LoadableData = (unsigned char *)malloc(m_File_size);
    if (m_LoadableData == nullptr)
    {
        DPRINTF("Cannot Allocate memory for loadable");
        exit(1);
    }

    actually_read = fread(m_LoadableData, 1, m_File_size, fp);
    if (actually_read != m_File_size)
    {
        free(m_LoadableData);
        DPRINTF("Read wrong size");
        exit(1);
    }
    fclose(fp);
}

void cuDLAContextStandalone::initialize()
{
    // Create CUDLA device
    m_cudla_err = cudlaCreateDevice(0, &m_DevHandle, CUDLA_STANDALONE);
    CHECK_CUDLA_ERR(m_cudla_err, "create CUDLA device");

    // Load CUDLA module
    m_cudla_err = cudlaModuleLoadFromMemory(m_DevHandle, m_LoadableData, m_File_size, &m_ModuleHandle, 0);
    CHECK_CUDLA_ERR(m_cudla_err, "load cuDLA module from memory");

    // Get number of input tensors
    m_cudla_err = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_NUM_INPUT_TENSORS, &m_cudla_module_attribute);
    CHECK_CUDLA_ERR(m_cudla_err, "get number of input tensors");
    m_NumInputTensors = m_cudla_module_attribute.numInputTensors;
    DPRINTF("numInputTensors = %d\n", m_NumInputTensors);

    // Get number of output tensors
    m_cudla_err = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_NUM_OUTPUT_TENSORS, &m_cudla_module_attribute);
    CHECK_CUDLA_ERR(m_cudla_err, "get number of output tensors");
    m_NumOutputTensors = m_cudla_module_attribute.numOutputTensors;
    DPRINTF("numOutputTensors = %d\n", m_NumOutputTensors);

    m_InputsBufContext.resize(m_NumInputTensors);
    m_OutputsBufContext.resize(m_NumOutputTensors);
    m_Input_Tensor_Descs.resize(m_NumInputTensors);
    m_Output_Tensor_Descs.resize(m_NumOutputTensors);
    m_InputBufRegPtrs.resize(m_NumInputTensors);
    m_OutputBufRegPtrs.resize(m_NumOutputTensors);

    m_cudla_module_attribute.inputTensorDesc = m_Input_Tensor_Descs.data();
    // Get IO Tensor Descriptor
    m_cudla_err = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &m_cudla_module_attribute);
    CHECK_CUDLA_ERR(m_cudla_err, "get input tensor descriptor");

    m_cudla_module_attribute.outputTensorDesc = m_Output_Tensor_Descs.data();
    m_cudla_err = cudlaModuleGetAttributes(m_ModuleHandle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &m_cudla_module_attribute);
    CHECK_CUDLA_ERR(m_cudla_err, "get output tensor descriptor");

    DPRINTF("Printing inputs tensor descriptor\n");
    for (uint32_t i = 0; i < m_NumInputTensors; i++)
    {
        m_InputsBufContext[i].cudla_tensor_desc = &m_Input_Tensor_Descs[i];
    }
    DPRINTF("Printing output tensor descriptor\n");
    for (uint32_t i = 0; i < m_NumOutputTensors; i++)
    {
        m_OutputsBufContext[i].cudla_tensor_desc = &m_Output_Tensor_Descs[i];
    }

    m_nvsci_err = NvSciBufModuleOpen(&m_NvSciBufModule);
    CHECK_NVSCI_ERR(m_nvsci_err, "open NvSci buffer module");

    // Create, set and reconcile NvSci attribute list.
    // Import memory and get cuda and cudla registered pointer.
    auto processNvSciBuf = [&](std::vector<NvSciBufferContext> &contexts) -> void {
        for (auto &context : contexts)
        {
            DPRINTF("-------------------------------------------\n");
            printTensorDesc(context.cudla_tensor_desc);
            createAndSetAttrList(m_NvSciBufModule, context.cudla_tensor_desc->size, &(context.unreconciled_attr_list));
            m_nvsci_err = NvSciBufAttrListReconcile(&(context.unreconciled_attr_list), 1,
                                                    &(context.reconciled_attr_list), &(context.conflict_list));
            CHECK_NVSCI_ERR(m_nvsci_err, "reconcile NvSciBuf attribute list");
            m_nvsci_err = NvSciBufObjAlloc(context.reconciled_attr_list, &(context.buf_obj));
            CHECK_NVSCI_ERR(m_nvsci_err, "alloc NvSciBuf Obj");

            memset(&(context.cudla_ext_mem_desc), 0, sizeof(context.cudla_ext_mem_desc));
            context.cudla_ext_mem_desc.extBufObject = (void *)context.buf_obj;
            context.cudla_ext_mem_desc.size         = context.cudla_tensor_desc->size;
            m_cudla_err = cudlaImportExternalMemory(m_DevHandle, &(context.cudla_ext_mem_desc),
                                                    &(context.buf_registered_ptr_cudla), 0);
            CHECK_CUDLA_ERR(m_cudla_err, "import memory to cudla");

            memset(&(context.cuda_mem_handle_desc), 0, sizeof(context.cuda_mem_handle_desc));
            context.cuda_mem_handle_desc.type                  = cudaExternalMemoryHandleTypeNvSciBuf;
            context.cuda_mem_handle_desc.handle.nvSciBufObject = context.buf_obj;
            context.cuda_mem_handle_desc.size                  = context.cudla_tensor_desc->size;
            m_cuda_err = cudaImportExternalMemory(&(context.ext_mem_raw_buf), &(context.cuda_mem_handle_desc));
            CHECK_CUDA_ERR(m_cuda_err, "import external memory to cuda");

            memset(&(context.cuda_ext_buffer_desc), 0, sizeof(context.cuda_ext_buffer_desc));
            context.cuda_ext_buffer_desc.offset = 0;
            context.cuda_ext_buffer_desc.size   = context.cudla_tensor_desc->size;
            m_cuda_err = cudaExternalMemoryGetMappedBuffer(&(context.buf_gpu), context.ext_mem_raw_buf,
                                                           &(context.cuda_ext_buffer_desc));
            CHECK_CUDA_ERR(m_cuda_err, "map external memory to cuda buffer");
            DPRINTF("-------------------------------------------\n");
        }
    };
    processNvSciBuf(m_InputsBufContext);
    processNvSciBuf(m_OutputsBufContext);

    // Create vector for registered cudla buffer pointer, this is for cudlaSubmitTask.
    auto createRegPtrVec = [&](std::vector<uint64_t *> &ptrs, const std::vector<NvSciBufferContext> &contexts) {
        for (size_t i = 0; i < contexts.size(); i++)
        {
            ptrs[i] = contexts[i].buf_registered_ptr_cudla;
        }
    };
    createRegPtrVec(m_InputBufRegPtrs, m_InputsBufContext);
    createRegPtrVec(m_OutputBufRegPtrs, m_OutputsBufContext);

    // Create NvSci sync module
    m_nvsci_err = NvSciSyncModuleOpen(&m_NvSciSyncModule);
    CHECK_NVSCI_ERR(m_nvsci_err, "create NvSci sync module");

    // Create -> Setup -> Reconcile -> Alloc sync object for waiter
    m_nvsci_err = NvSciSyncAttrListCreate(m_NvSciSyncModule, &(m_WaitEventContext.waiter_attr_list));
    CHECK_NVSCI_ERR(m_nvsci_err, "create NvSci waiter attr list");
    m_nvsci_err = NvSciSyncAttrListCreate(m_NvSciSyncModule, &(m_WaitEventContext.signaler_attr_list));
    CHECK_NVSCI_ERR(m_nvsci_err, "create NvSci signal attr list");
    m_cudla_err = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t *>(m_WaitEventContext.waiter_attr_list),
                                              CUDLA_NVSCISYNC_ATTR_WAIT);
    CHECK_CUDLA_ERR(m_cudla_err, "get NvSci waiter sync attributes");
    m_cuda_err = cudaDeviceGetNvSciSyncAttributes(m_WaitEventContext.signaler_attr_list, 0, cudaNvSciSyncAttrSignal);
    CHECK_CUDA_ERR(m_cuda_err, "cuda get NvSci signal list");
    NvSciSyncAttrList wait_event_attrs[2] = {m_WaitEventContext.signaler_attr_list,
                                             m_WaitEventContext.waiter_attr_list};
    m_nvsci_err = NvSciSyncAttrListReconcile(wait_event_attrs, 2, &(m_WaitEventContext.reconciled_attr_list),
                                             &(m_WaitEventContext.conflict_list));
    CHECK_NVSCI_ERR(m_nvsci_err, "reconciled NvSci sync attr list");
    m_nvsci_err = NvSciSyncObjAlloc(m_WaitEventContext.reconciled_attr_list, &(m_WaitEventContext.sync_obj));
    CHECK_NVSCI_ERR(m_nvsci_err, "allocate NvSci sync object");

    // Import semaphore to cudla
    memset(&m_WaitEventContext.cudla_ext_sema_mem_desc, 0, sizeof(m_WaitEventContext.cudla_ext_sema_mem_desc));
    m_WaitEventContext.cudla_ext_sema_mem_desc.extSyncObject = m_WaitEventContext.sync_obj;
    m_cudla_err = cudlaImportExternalSemaphore(m_DevHandle, &(m_WaitEventContext.cudla_ext_sema_mem_desc),
                                               &m_WaitEventContext.nvsci_sync_obj_reg_ptr, 0);
    CHECK_CUDLA_ERR(m_cudla_err, "cudla import external semaphore");

    // Create wait event, this is for cudlaSubmitTask
    m_WaitEvents                              = (cudlaWaitEvents *)malloc(sizeof(cudlaWaitEvents));
    m_WaitEvents->numEvents                   = 1;
    m_WaitEventContext.nvsci_fence_ptr        = (NvSciSyncFence *)calloc(1, sizeof(NvSciSyncFence));
    m_WaitEventContext.cudla_fence_ptr        = (CudlaFence *)malloc(sizeof(CudlaFence));
    m_WaitEventContext.cudla_fence_ptr->fence = m_WaitEventContext.nvsci_fence_ptr;
    m_WaitEventContext.cudla_fence_ptr->type  = CUDLA_NVSCISYNC_FENCE;
    m_WaitEvents->preFences                   = m_WaitEventContext.cudla_fence_ptr;

    // Import semaphore to cuda
    cudaExternalSemaphoreHandleDesc signalExtSemDesc;
    memset(&signalExtSemDesc, 0, sizeof(signalExtSemDesc));
    signalExtSemDesc.type                = cudaExternalSemaphoreHandleTypeNvSciSync;
    signalExtSemDesc.handle.nvSciSyncObj = (void *)(m_WaitEventContext.sync_obj);
    m_cuda_err                           = cudaImportExternalSemaphore(&m_SignalSem, &signalExtSemDesc);
    memset(&m_SignalParams, 0, sizeof(m_SignalParams));
    m_SignalParams.params.nvSciSync.fence = (void *)(m_WaitEventContext.nvsci_fence_ptr);
    m_SignalParams.flags                  = 0;

    // Create -> Setup -> Reconcile -> Alloc sync object for signaler
    m_nvsci_err = NvSciSyncAttrListCreate(m_NvSciSyncModule, &(m_SignalEventContext.signaler_attr_list));
    CHECK_NVSCI_ERR(m_nvsci_err, "create NvSci signal attr list");
    m_nvsci_err = NvSciSyncAttrListCreate(m_NvSciSyncModule, &(m_SignalEventContext.waiter_attr_list));
    CHECK_NVSCI_ERR(m_nvsci_err, "create NvSci waiter attr list");
    m_cudla_err = cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t *>(m_SignalEventContext.signaler_attr_list),
                                              CUDLA_NVSCISYNC_ATTR_SIGNAL);
    CHECK_CUDLA_ERR(m_cudla_err, "get NvSci sync attributes");
    m_cuda_err = cudaDeviceGetNvSciSyncAttributes(m_SignalEventContext.waiter_attr_list, 0, cudaNvSciSyncAttrWait);
    CHECK_CUDA_ERR(m_cuda_err, "cuda get NvSci wait attribute list");
    NvSciSyncAttrList signal_event_attrs[2] = {m_SignalEventContext.signaler_attr_list,
                                               m_SignalEventContext.waiter_attr_list};
    m_nvsci_err = NvSciSyncAttrListReconcile(signal_event_attrs, 2, &(m_SignalEventContext.reconciled_attr_list),
                                             &(m_SignalEventContext.conflict_list));
    CHECK_NVSCI_ERR(m_nvsci_err, "reconciled NvSci sync attr list");
    m_nvsci_err = NvSciSyncObjAlloc(m_SignalEventContext.reconciled_attr_list, &(m_SignalEventContext.sync_obj));
    CHECK_NVSCI_ERR(m_nvsci_err, "allocate NvSci sync object");

    // Import semaphore to cudla
    memset(&m_SignalEventContext.cudla_ext_sema_mem_desc, 0, sizeof(m_SignalEventContext.cudla_ext_sema_mem_desc));
    m_SignalEventContext.cudla_ext_sema_mem_desc.extSyncObject = m_SignalEventContext.sync_obj;
    m_cudla_err = cudlaImportExternalSemaphore(m_DevHandle, &(m_SignalEventContext.cudla_ext_sema_mem_desc),
                                               &m_SignalEventContext.nvsci_sync_obj_reg_ptr, 0);
    CHECK_CUDLA_ERR(m_cudla_err, "cudla import external semaphore");

    // Create signal event, this is for cudlaSubmitTask
    m_SignalEvents                              = (cudlaSignalEvents *)malloc(sizeof(cudlaSignalEvents));
    m_SignalEvents->numEvents                   = 1;
    m_SignalEvents->devPtrs                     = &(m_SignalEventContext.nvsci_sync_obj_reg_ptr);
    m_SignalEventContext.nvsci_fence_ptr        = (NvSciSyncFence *)calloc(1, sizeof(NvSciSyncFence));
    m_SignalEventContext.cudla_fence_ptr        = (CudlaFence *)malloc(sizeof(CudlaFence));
    m_SignalEventContext.cudla_fence_ptr->fence = m_SignalEventContext.nvsci_fence_ptr;
    m_SignalEventContext.cudla_fence_ptr->type  = CUDLA_NVSCISYNC_FENCE;
    m_SignalEvents->eofFences                   = m_SignalEventContext.cudla_fence_ptr;

    // Import semaphore to cuda
    cudaExternalSemaphoreHandleDesc waiterExtSemDesc;
    memset(&waiterExtSemDesc, 0, sizeof(waiterExtSemDesc));
    waiterExtSemDesc.type                = cudaExternalSemaphoreHandleTypeNvSciSync;
    waiterExtSemDesc.handle.nvSciSyncObj = (void *)(m_SignalEventContext.sync_obj);
    m_cuda_err                           = cudaImportExternalSemaphore(&m_WaitSem, &waiterExtSemDesc);
    CHECK_CUDA_ERR(m_cuda_err, "import external semaphore to cuda");
    memset(&m_WaitParams, 0, sizeof(m_WaitParams));
    m_WaitParams.params.nvSciSync.fence = (void *)m_SignalEventContext.nvsci_fence_ptr;
    m_WaitParams.flags                  = 0;

    // Init cudla task
    m_Task.moduleHandle     = m_ModuleHandle;
    m_Task.numInputTensors  = m_NumInputTensors;
    m_Task.numOutputTensors = m_NumOutputTensors;
    m_Task.inputTensor      = m_InputBufRegPtrs.data();
    m_Task.outputTensor     = m_OutputBufRegPtrs.data();
    m_Task.waitEvents       = m_WaitEvents;
    m_Task.signalEvents     = m_SignalEvents;
}

uint64_t cuDLAContextStandalone::getInputTensorSizeWithIndex(int32_t index) { return m_Input_Tensor_Descs[index].size; }

uint64_t cuDLAContextStandalone::getOutputTensorSizeWithIndex(int32_t index)
{
    return m_Output_Tensor_Descs[index].size;
}

void *cuDLAContextStandalone::getInputCudaBufferPtr(int32_t index) { return m_InputsBufContext[index].buf_gpu; }
void *cuDLAContextStandalone::getOutputCudaBufferPtr(int32_t index) { return m_OutputsBufContext[index].buf_gpu; }

uint32_t cuDLAContextStandalone::getNumInputTensors() { return m_NumInputTensors; }

uint32_t cuDLAContextStandalone::getNumOutputTensors() { return m_NumOutputTensors; }

int cuDLAContextStandalone::submitDLATask(cudaStream_t streamToRun)
{
    m_cuda_err = cudaSignalExternalSemaphoresAsync(&m_SignalSem, &m_SignalParams, 1, streamToRun);
    CHECK_CUDA_ERR(m_cuda_err, "signal external semaphores on previous stream");
    m_cudla_err = cudlaSubmitTask(m_DevHandle, &m_Task, 1, nullptr, 0);
    CHECK_CUDLA_ERR(m_cudla_err, "submit cudla task");
    m_cuda_err = cudaWaitExternalSemaphoresAsync(&m_WaitSem, &m_WaitParams, 1, streamToRun);
    CHECK_CUDA_ERR(m_cuda_err, "wait external semaphores on previous stream");
    return 0;
}

void cuDLAContextStandalone::releaseNvSciBufferContexts(std::vector<NvSciBufferContext> &contexts)
{
    // release resource in reverse construct order, good practice but may not always be necessary
    for (auto &context : contexts)
    {
        m_cuda_err = cudaDestroyExternalMemory(context.ext_mem_raw_buf);
        CHECK_CUDA_ERR(m_cuda_err, "destroy external cuda memory");
        context.buf_gpu = nullptr;
        m_cudla_err     = cudlaMemUnregister(m_DevHandle, context.buf_registered_ptr_cudla);
        CHECK_CUDLA_ERR(m_cudla_err, "unregister cudla buffer ptr");
        NvSciBufObjFree(context.buf_obj);
        NvSciBufAttrListFree(context.conflict_list);
        NvSciBufAttrListFree(context.reconciled_attr_list);
        NvSciBufAttrListFree(context.unreconciled_attr_list);
    }
}

void cuDLAContextStandalone::releaseNvSciSyncContext(NvSciSyncContext &context)
{
    // release resource in reverse construct order, good practice but may not always be necessary
    free(context.cudla_fence_ptr);
    context.cudla_fence_ptr = nullptr;
    NvSciSyncFenceClear(context.nvsci_fence_ptr);
    context.nvsci_fence_ptr = nullptr;
    m_cudla_err             = cudlaMemUnregister(m_DevHandle, context.nvsci_sync_obj_reg_ptr);
    CHECK_CUDLA_ERR(m_cudla_err, "unregister cudla sync ptr");
    NvSciSyncObjFree(context.sync_obj);
    NvSciSyncAttrListFree(context.conflict_list);
    NvSciSyncAttrListFree(context.reconciled_attr_list);
    NvSciSyncAttrListFree(context.waiter_attr_list);
    NvSciSyncAttrListFree(context.signaler_attr_list);
}

cuDLAContextStandalone::~cuDLAContextStandalone()
{
    m_cuda_err = cudaDestroyExternalSemaphore(m_WaitSem);
    CHECK_CUDA_ERR(m_cuda_err, "destroy external cuda wait semaphore");
    m_cuda_err = cudaDestroyExternalSemaphore(m_SignalSem);
    CHECK_CUDA_ERR(m_cuda_err, "destroy external cuda signal semaphore");
    releaseNvSciBufferContexts(m_InputsBufContext);
    releaseNvSciBufferContexts(m_OutputsBufContext);
    releaseNvSciSyncContext(m_WaitEventContext);
    releaseNvSciSyncContext(m_SignalEventContext);
    free(m_WaitEvents);
    m_WaitEvents = nullptr;
    free(m_SignalEvents);
    m_SignalEvents = nullptr;
    NvSciBufModuleClose(m_NvSciBufModule);
    NvSciSyncModuleClose(m_NvSciSyncModule);
    m_cudla_err = cudlaModuleUnload(m_ModuleHandle, 0);
    CHECK_CUDLA_ERR(m_cudla_err, "unload cudla module");
    m_ModuleHandle = nullptr;
    m_cudla_err    = cudlaDestroyDevice(m_DevHandle);
    CHECK_CUDLA_ERR(m_cudla_err, "unload cudla device handle");
    m_DevHandle = nullptr;
    free(m_LoadableData);
    m_LoadableData = nullptr;
}
