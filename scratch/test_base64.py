import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt

# Simulate the plot generation
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 3, 5, 7, 9])
predictions = np.array([1.6, 3.4, 5.2, 7.0, 8.8])

plt.figure(figsize=(6,4))
plt.scatter(X, Y, color="blue", label="Actual Data")
plt.plot(X, predictions, color="red", linewidth=2, label="Best Fit Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()

buf = io.BytesIO()
plt.savefig(buf, format="png")
buf.seek(0)
img_bytes = buf.read()
print("Bytes size:", len(img_bytes))
regression_base64 = base64.b64encode(img_bytes).decode("utf-8")
print("Base64 string size:", len(regression_base64))
print("Start of base64:", regression_base64[:50])

# Decode and write to a test file to verify
with open("test_out.png", "wb") as f:
    f.write(base64.b64decode(regression_base64))
print("Wrote test_out.png successfully")
