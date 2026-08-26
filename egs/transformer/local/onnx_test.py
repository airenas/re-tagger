import onnxruntime as ort

session = ort.InferenceSession(
    "models/model.onnx",
    providers=["CPUExecutionProvider"],
)

print(session.get_inputs())
print(session.get_outputs())