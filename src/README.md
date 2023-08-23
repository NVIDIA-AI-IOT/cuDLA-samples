## Pipeline

The pipeline in this sample is as below:

| Steps                               | Backend | Time  |
| ----------------------------------- | ------- | ----- |
| OpenCV load image + preprocessing   | CPU     | N/A   |
| MatX reformat to cuDLA input format | GPU     | 0.7ms |
| cuDLA inference                     | DLA     | 4.5ms |
| MatX reformat to FP16 planar format | GPU     | 0.7ms |
| Post-process(decode_box, nms)  | GPU     | 0.1-0.2ms   |
| Get final bbox data                 | CPU     | N/A   |

## Code Structure

```
.
├── cudla_context_hybrid.cpp            # cuDLA inference context in hybrid mode [cuDLA inference]
├── cudla_context_hybrid.h              # cuDLA inference context in hybrid mode [cuDLA inference]
├── cudla_context_standalone.cpp # cuDLA inference context in standalone mode [cuDLA inference]
├── cudla_context_standalone.h   # cuDLA inference context in standalone mode [cuDLA inference]
├── decode_nms.cu                # Decode bbox and NMS [post-processing]
├── decode_nms.h                 # Decode bbox and NMS [post-processing]
├── matx_reformat                # data reformat for cuDLA inputs/outputs [pre-processing] and [post-processing]
│   ├── build_matx_reformat.sh
│   ├── CMakeLists.txt
│   ├── MatX
│   ├── matx_reformat.cu
│   ├── matx_reformat.h
│   ├── README.md
│   └── test.cpp
├── validate_coco.cpp            # Validation app, main function [jpg --> OpenCV decode]
├── yolov5.cpp                   # Yolov5 pipeline [pre-processing --> Inference(cuDLA) --> post-processing]
└── yolov5.h                     # Yolov5 pipeline [pre-processing --> Inference(cuDLA) --> post-processing]
```
