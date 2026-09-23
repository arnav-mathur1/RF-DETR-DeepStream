# RF-DETR in NVIDIA DeepStream and TensorRT Inference

This was an important part of my work in Summer 2026. There were two main parts:
- Part 1: Converting RF-DETR TensorRT outputs into standard DeepStream detections with a custom C++ parser
- Part 2: Matching RF-DETR's training-time normalization when performing inference through DeepStream and TensorRT

# Aside

I worked on the C++ parser about 1-2 months before a developer at RF-DETR wrote a blog working on something similar. You can read along [here](https://blog.roboflow.com/rf-detr-nvidia-deepstream/).

For my implementation, I based it on RF-DETR's exported tensor format and [post-processing logic](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/postprocess.py) from the official Roboflow RF-DETR implementation.

This process helped greatly in debugging and dealing with the second part.

# RF-DETR DeepStream Output Parser

DeepStream does not know how to interpret RF-DETR's output tensors. The exported model produces two main outputs:
- dets --- [num_queries, 4]
- labels --- [num_queries, num_classes]

- dets contains normalized bounding boxes in: (cx, cy, width, height)
- labels contains the class logits associated with each query

The custom parser in "nvdsparsebbox_RFDETR.cpp" converts these tensors into NvDsInferParseObjectInfo objects that DeepStream can use.

The parser:
- locates the dets and labels output tensors
- supports FP32 and FP16 outputs
- reads each RF-DETR object query
- selects the highest-scoring class
- applies sigmoid to obtain a confidence score
- applies DeepStream per-class confidence thresholds
- converts normalized (cx, cy, w, h) boxes into pixel coordinates
- clips boxes to the input image boundaries
- returns the resulting detections to DeepStream
- keeps at most the top 300 detections by confidence

## DeepStream Configuration

The parser is referenced from the DeepStream `nvinfer` configuration.

Example:

```ini
[property]

onnx-file=/path/to/inference_model_normalized_fp32.onnx

infer-dims=3;672;672
batch-size=1
network-mode=2

custom-lib-path=/path/to/libnvdsinfer_custom_impl_RFDETR.so
parse-bbox-func-name=NvDsInferParseRFDETR

cluster-mode=4
```

Per-class detection thresholds can then be configured using the normal DeepStream configuration.

Example:

```ini
[class-attrs-all]
pre-cluster-threshold=0.30
```

# RF-DETR Input Normalization for TensorRT

The second issue is input preprocessing. 

RF-DETR expects ImageNet-normalized RGB input. Incorrect preprocessing can cause detections to become inaccurate or disappear entirely.

## Baking Normalization Into the ONNX Model

`bake_normalization.py` modifies the exported ONNX graph.

The original graph begins approximately as:

```text
input
  ↓
RF-DETR network
```

The script inserts two operations:

```text
input
  ↓
Sub(mean)
  ↓
Div(std)
  ↓
RF-DETR network
```

The only difference is that the original network now receives normalized data.

## Running the Normalization Script

Install the required Python packages:

```bash
pip install onnx numpy
```

Then run:

```bash
python3 bake_normalization.py
```

The script takes the raw RF-DETR ONNX model and produces a second ONNX model with the per-channel normalization operations inserted.

# Building a TensorRT Engine

DeepStream can build and cache the TensorRT engine automatically from the ONNX file.

Alternatively, the model can be tested independently with TensorRT's `trtexec`.

Example:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=inference_model_normalized_fp32.onnx \
    --saveEngine=rfdetr_fp16.engine \
    --fp16
```

For a fixed input shape:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=inference_model_normalized_fp32.onnx \
    --saveEngine=rfdetr_fp16.engine \
    --fp16 \
    --shapes=input:1x3x672x672
```

Exact paths may differ depending on the TensorRT installation.

The generated engine can then be referenced directly from DeepStream if desired.
