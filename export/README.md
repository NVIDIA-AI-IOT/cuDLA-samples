

# **Setup**
## 1. Clone and apply patch

### Git clone [yolov5](https://github.com/ultralytics/yolov5) and install Dependencies, Please refer [INSTALL](https://docs.ultralytics.com/yolov5/quickstart_tutorial/) 
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
git checkout v7.0
```

### Apply this patch to your yolov5 project
```bash
cp -r  export/yolov5-qat/* yolov5/
```
## 2. Install dependencies
```bash
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
cd  export/qdq_translator
pip install -r requirements.txt
```
We use [TensorRT's pytorch quntization tool](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) to fine-tune yolov5 from the pre-trained weight. Here is the reference link for [TensorRT's approach to quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#intro-quantization).
# **Yolov5 QAT Fine-tuning and Export**
We are proposing two possible options for Q/DQ node insertion for YOLOV5 QAT. Both methods have their advantages, and we have implemented support for both in this project.

## **Option#1**
Place Q/DQ nodes as recommended in [TensorRT Processing of Q/DQ Networks](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#tensorrt-process-qdq). This method complies with TensorRT's fusion strategy for Q/DQ layers.These strategies are mostly intended for GPU inference. For compatibility with DLA, missing Q/DQ nodes can be derived using the scales from their neighboring layers in the Q/DQ Translator. 

- QAT fine-tuning

Replace cocodir with your own path.
 ```bash
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
python scripts/qat.py quantize yolov5s.pt --ptq=ptq.pt --qat=qat.pt --cocodir=datasets/coco --eval-ptq --eval-origin 
```

- Export QAT model

To test the mean Average Precision (mAP) of the model, set the size to 672. Otherwise, set the size to 640.

It is necessary to set the flag **--noanchor** when exporting to ONNX to ensure that the exported model does not include anchor nodes. The computation of the anchor nodes will be implemented on CUDA to achieve better inference performance.

```bash
python scripts/qat.py export qat.pt --size=672 --save=yolov5_trimmed_qat.onnx --dynamic --noanchor
```
- Convert QAT model to PTQ model and INT8 calibration cache

If the program throws an exception when checking the weights scales, you can adjust the value of rtol.
```bash
python export/qdq_translator/qdq_translator.py --input_onnx_models=yolov5_trimmed_qat.onnx --output_dir=data/model/ --infer_concat_scales --infer_mul_scales 
```

## **Option#2** 
Insert Q/DQ nodes at every layer to ensure all tensors have int8 scales. Compared to Option 1, all layers' scales can be obtained during model fine-tuning. But this method may potentially disrupt TensorRT's fusion strategy with Q/DQ layers if running inference on GPU. This is why importing an ONNX graph with Q/DQ nodes placed with Option 2, the latency may be higher on GPU with this approach.

- QAT fine-tuning
  
 ```bash
python scripts/qat.py quantize yolov5s.pt --ptq=ptq.pt --qat=qat.pt --cocodir=datasets/coco --eval-ptq --eval-origin --all-node-with-qdq
```
- Export QAT model

```bash
python scripts/qat.py export qat.pt --size=672 --save=yolov5_trimmed_qat.onnx --dynamic --noanchor
```
- Convert QAT model to PTQ model and INT8 calibration cache
```bash
python export/qdq_translator/qdq_translator.py --input_onnx_models=yolov5_trimmed_qat.onnx --output_dir=data/model/ 
```
## **Notes**
As confirmed by experimental validation, the YOLOv5 model was verified on the COCO 2017 val dataset with a resolution of 672x672, Option 1 and Option 2 respectively achieved mAP scores of 37.1 and 37.0.

