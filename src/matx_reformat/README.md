#### Build and test

You need at least cmake 3.18+ to build this project.

MatX has it's own dependency, please refer to https://nvidia.github.io/MatX/build.html

aarch64
```
mkdir build && cd build
cmake ..
make clean && make
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
./test
```

#### how to integrate

Please refer to matx_reformat.h