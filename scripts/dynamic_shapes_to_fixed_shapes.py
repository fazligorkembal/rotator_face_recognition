import onnx
from onnx import shape_inference


model = onnx.load("/home/user/Documents/rfr/models/det_10g.onnx")

for input_tensor in model.graph.input:
    print(input_tensor.name, input_tensor.type.tensor_type.shape)

for input_tensor in model.graph.input:
    dim_proto = input_tensor.type.tensor_type.shape.dim
    dim_proto[0].dim_value = 1  # batch
    dim_proto[1].dim_value = 3  # channel
    dim_proto[2].dim_value = 640  # height
    dim_proto[3].dim_value = 640  # width

model = shape_inference.infer_shapes(model)

onnx.save(
    model, "/home/user/Documents/rfr/models/det_10g_fixed_input_1_batch.onnx"
)
