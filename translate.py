import torch

device = 'cuda'

model = torch.load("model.pt")
batch_size = 1
input_shape = (28*28, 512)

model.eval()

x = torch.randn(batch_size, *input_shape)
x = x.to(device)
export_onnx_file = "model.onnx"
torch.onnx.export(model, x,
                  export_onnx_file,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"},
                                "output": {0: "batch_size"}})
