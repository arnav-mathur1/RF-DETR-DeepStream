================================================================================
 RF-DETR (b4) INPUT NORMALIZATION — HOW IT'S DONE
================================================================================
Dir : /home/spark/Arnav/deepstream/DeepStream-Yolo/exports/export_rfdetr_small_672_total_b4_coco92_2
Files:
  inference_model.onnx                  <- RAW export, NO normalization baked in
  inference_model_normalized_fp32.onnx  <- normalization baked in (USE THIS one)

--------------------------------------------------------------------------------
1. WHY THIS EXISTS
--------------------------------------------------------------------------------
RF-DETR was trained on ImageNet-normalized input. The real preprocessing is:

    x_uint8  in [0,255]
    x01      = x_uint8 / 255.0                      # scale to [0,1]
    x_norm   = (x01 - mean) / std                   # per-channel ImageNet norm
        mean = [0.485, 0.456, 0.406]   (R,G,B)
        std  = [0.229, 0.224, 0.225]   (R,G,B)

The network's first Conv expects x_norm. If you feed it anything else, detections
are wrong / missing (this was the known "missing /std normalize" accuracy gap).

DeepStream nvinfer can only apply ONE scalar formula per tensor:

    y = net-scale-factor * (x - offset)

That gives the SAME scale to all 3 channels. ImageNet std differs per channel
(0.229 vs 0.224 vs 0.225), so a single scalar CANNOT reproduce (x01 - mean)/std
exactly. Solution: split the work.

    DeepStream  does:   x01   = (1/255) * x_uint8         (net-scale-factor=1/255)
    ONNX graph  does:   x_norm = (x01 - mean) / std        (baked-in Sub + Div)

So we "bake" the per-channel (x - mean)/std step into the ONNX itself.

--------------------------------------------------------------------------------
2. WHAT "BAKING" MEANS (graph surgery)
--------------------------------------------------------------------------------
The raw export's input feeds straight into the patch-embed path:

    input[4,3,672,672] --> Cast --> Conv --> ... (rest of network)

We insert two ops right after the input so every original consumer now receives
NORMALIZED data instead:

    input[4,3,672,672]
        --> Sub(input, norm_mean)  -> norm_sub_out      # x01 - mean
        --> Div(norm_sub_out, norm_std) -> norm_div_out  # / std
        --> Cast --> Conv --> ... (rest of network, unchanged)

    norm_mean = [0.485,0.456,0.406]  shape [1,3,1,1] fp32   (broadcasts over H,W)
    norm_std  = [0.229,0.224,0.225]  shape [1,3,1,1] fp32

The external input tensor is still named "input" and still [4,3,672,672]; it now
expects values in [0,1] (because DeepStream pre-scales by 1/255). This is byte-for
-byte identical to how the b1 inference_model_normalized_fp32.onnx was produced.

--------------------------------------------------------------------------------
3. THE COMMAND / SCRIPT (reproducible)
--------------------------------------------------------------------------------
Run from: /home/spark/Arnav/deepstream/DeepStream-Yolo
Needs only: python3 with `onnx` and `numpy` (no GPU, no checkpoint, no retrain).

----------------------------- bake_normalization.py ----------------------------
import onnx, numpy as np
from onnx import helper, numpy_helper

SRC = "exports/export_rfdetr_small_672_total_b4_coco92_2/inference_model.onnx"
DST = "exports/export_rfdetr_small_672_total_b4_coco92_2/inference_model_normalized_fp32.onnx"

m = onnx.load(SRC)
g = m.graph
assert g.input[0].name == "input", g.input[0].name

# ImageNet constants as [1,3,1,1] fp32 initializers
mean = numpy_helper.from_array(np.array([0.485,0.456,0.406], np.float32).reshape(1,3,1,1), "norm_mean")
std  = numpy_helper.from_array(np.array([0.229,0.224,0.225], np.float32).reshape(1,3,1,1), "norm_std")
g.initializer.extend([mean, std])

# Sub then Div
sub = helper.make_node("Sub", ["input","norm_mean"],      ["norm_sub_out"], name="norm_sub")
div = helper.make_node("Div", ["norm_sub_out","norm_std"], ["norm_div_out"], name="norm_div")

# reroute every original consumer of "input" to the normalized tensor
rewired = 0
for node in g.node:
    for i,t in enumerate(node.input):
        if t == "input":
            node.input[i] = "norm_div_out"; rewired += 1

# prepend the two nodes (order matters: Sub must run before Div)
g.node.insert(0, div)
g.node.insert(0, sub)

onnx.checker.check_model(m)
onnx.save(m, DST)
print(f"rewired {rewired} consumer(s) -> {DST}")
--------------------------------------------------------------------------------

Run it:
    cd /home/spark/Arnav/deepstream/DeepStream-Yolo
    python3 bake_normalization.py

Expected output:
    rewired 1 consumer(s) -> exports/.../inference_model_normalized_fp32.onnx

--------------------------------------------------------------------------------
4. VERIFY IT WORKED
--------------------------------------------------------------------------------
    python3 - <<'PY'
    import onnx
    g = onnx.load("exports/export_rfdetr_small_672_total_b4_coco92_2/inference_model_normalized_fp32.onnx",
                  load_external_data=False).graph
    print("input:", g.input[0].name,
          [d.dim_value for d in g.input[0].type.tensor_type.shape.dim])
    print("first 2 nodes:", [(n.op_type, list(n.input), list(n.output)) for n in g.node[:2]])
    PY

Expected:
    input: input [4, 3, 672, 672]
    first 2 nodes: [('Sub', ['input','norm_mean'], ['norm_sub_out']),
                    ('Div', ['norm_sub_out','norm_std'], ['norm_div_out'])]

--------------------------------------------------------------------------------
5. HOW DEEPSTREAM IS CONFIGURED TO MATCH (the other half)
--------------------------------------------------------------------------------
In config_infer_primary_uav_video_coco92_parser2.txt :

    net-scale-factor=0.00392156862745098   # = 1/255  -> DeepStream makes x01=[0,1]
    model-color-format=0                    # RGB (matches mean/std channel order)
    # offsets are omitted (= 0) because the (x-mean) part is done INSIDE the onnx
    onnx-file=.../export_rfdetr_small_672_total_b4_coco92_2/inference_model_normalized_fp32.onnx
    infer-dims=3;672;672
    batch-size=4
    network-mode=2                          # FP16 engine built from this onnx

Pipeline summary:
    camera frame (uint8 RGB)
      -> DeepStream resize 672x672 (scaling-filter=3 super-sample, antialiased)
      -> DeepStream net-scale 1/255            => x01 in [0,1]
      -> ONNX Sub/Div (baked)                  => (x01-mean)/std  == training input
      -> RF-DETR backbone ... -> dets/labels
      -> NvDsInferParseRFDETRPy (top-300, NMS-free)

IMPORTANT: only use inference_model_normalized_fp32.onnx in the config. If you
point onnx-file at the raw inference_model.onnx, the (x-mean)/std step is missing
and detections will be wrong. Do NOT add net-scale offsets to "compensate" — a
scalar scale cannot reproduce per-channel std; that is exactly why we bake it.

================================================================================
