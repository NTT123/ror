"""

Install:

pip install onnxruntime
"""


import numpy as np
import onnxruntime as ort

x = np.zeros((1, 1, 28, 28)).astype(np.float32)
ort_sess = ort.InferenceSession("models/mnist.onnx")
outputs = ort_sess.run(None, {"Input3": x})
print("zero input", outputs[0][0])



x = np.ones((1, 1, 28, 28)).astype(np.float32)
ort_sess = ort.InferenceSession("models/mnist.onnx")
outputs = ort_sess.run(None, {"Input3": x})
print("one input", outputs[0][0].tolist())
