## File structure

```
.
├── build_dla_standalone_loadable.sh     # scripts to build loadable, better performance.
├── build_dla_standalone_loadable_v2.sh  # scripts to build loadable, better accuracy.
├── qat2ptq.cache                        # calibration cache from QAT model
├── README.md
├── yolov5s_trimmed_reshape_tranpose.onnx # onnx for DLA FP16
└── yolov5_trimmed_qat.onnx               # onnx for DLA INT8
```