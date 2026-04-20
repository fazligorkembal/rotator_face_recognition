import onnx

model = onnx.load('/home/user/Documents/rfr/models/det_10g.onnx')
model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch'
onnx.save(model, '/home/user/Documents/rfr/models/det_10g_dynamic.onnx')
print('Done')
