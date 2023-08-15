#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

echo "Build DLA loadable for fp16 and int8"
mkdir -p data/loadable
TRTEXEC=/usr/src/tensorrt/bin/trtexec
${TRTEXEC} --onnx=data/model/yolov5s_trimmed_reshape_tranpose.onnx --verbose --fp16 --saveEngine=data/loadable/yolov5.fp16.fp16chw16in.fp16chw16out.standalone.bin --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0
${TRTEXEC} --minShapes=images:1x3x672x672 --maxShapes=images:1x3x672x672 --optShapes=images:1x3x672x672 --shapes=images:1x3x672x672 --onnx=data/model/yolov5_trimmed_qat.onnx --useDLACore=0 --buildDLAStandalone --saveEngine=data/loadable/yolov5.int8.int8hwc4in.fp16chw16out.standalone.bin  --inputIOFormats=int8:dla_hwc4 --outputIOFormats=fp16:chw16 --int8 --fp16 --calib=data/model/qat2ptq.cache --precisionConstraints=obey --layerPrecisions="/model.24/m.0/Conv":fp16,"/model.24/m.1/Conv":fp16,"/model.24/m.2/Conv":fp16,"/model.23/cv3/conv/Conv":fp16,"/model.23/cv3/act/Sigmoid":fp16,"/model.23/cv3/act/Mul":fp16
