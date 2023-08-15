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

cd src/matx_reformat/

mkdir -p build
cd build

system_cmake_version=$(cmake --version | head -n1 | cut -d' ' -f3)
major_version=$(echo $system_cmake_version | cut -d'.' -f2)
if [ $major_version -ge 18 ]
then
    echo "System cmake version is 3.${major_version} which greater than 3.18. Using system cmake..."
    CMAKE=cmake
else
    echo "System cmake version is less than 3.18. Downloading CMake 3.25.3..."
    # Download and extract CMake 3.25.3
    wget https://github.com/Kitware/CMake/releases/download/v3.25.3/cmake-3.25.3-linux-aarch64.tar.gz
    tar xf cmake-3.25.3-linux-aarch64.tar.gz
    CMAKE=./cmake-3.25.3-linux-aarch64/bin/cmake
fi

${CMAKE} ..
make clean && make
