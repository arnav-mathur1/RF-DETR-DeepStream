import onnx, numpy as np
from onnx import helper, numpy_helper

# Bake ImageNet (x-mean)/std normalization into the raw b6 672 export, exactly like
# the 672 b4/b1 inference_model_normalized_fp32.onnx were produced.
d = "exports/export_rfdetr_small_672_total_b6_coco92_2"
SRC = f"{d}/inference_model.onnx"
DST = f"{d}/inference_model_normalized_fp32.onnx"

m = onnx.load(SRC)
g = m.graph
assert g.input[0].name == "input", g.input[0].name

# ImageNet constants as [1,3,1,1] fp32 initializers
mean = numpy_helper.from_array(np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1), "norm_mean")
std  = numpy_helper.from_array(np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1), "norm_std")
g.initializer.extend([mean, std])

# Sub then Div
sub = helper.make_node("Sub", ["input", "norm_mean"],       ["norm_sub_out"], name="norm_sub")
div = helper.make_node("Div", ["norm_sub_out", "norm_std"], ["norm_div_out"], name="norm_div")

# reroute every original consumer of "input" to the normalized tensor
rewired = 0
for node in g.node:
    for i, t in enumerate(node.input):
        if t == "input":
            node.input[i] = "norm_div_out"; rewired += 1

# prepend the two nodes (order matters: Sub must run before Div)
g.node.insert(0, div)
g.node.insert(0, sub)

onnx.checker.check_model(m)
onnx.save(m, DST)
print(f"{d}: rewired {rewired} consumer(s) -> {DST}")
