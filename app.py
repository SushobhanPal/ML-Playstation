import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods =["POST"])
def predict():
    try:
        data = request.get_json()
        x_raw = data.get('x', '')
        y_raw = data.get('y', '')

        x = [float(i.strip()) for i in x_raw.split(",") if i.strip()]
        y = [float(i.strip()) for i in y_raw.split(",") if i.strip()]
    except (ValueError, TypeError, AttributeError):
        return jsonify({"error": "Inputs must be comma-separated lists of numbers."}), 400

    n = len(x)
    if n != len(y):
        return jsonify({"error": f"X and Y must have the same number of elements. (X has {n}, Y has {len(y)})"}), 400

    if n < 2:
        return jsonify({"error": "Linear regression requires at least 2 data points."}), 400

    sum_x = sum(x)
    sum_y = sum(y)

    xy = []
    x2 = []
    for i in range(n):
        xy.append(x[i]*y[i])
        x2.append(x[i]*x[i])

    sum_xy = sum(xy)
    sum_x2 = sum(x2)

    denominator = ((n*sum_x2) - (sum_x*sum_x))
    if denominator == 0:
        return jsonify({"error": "Cannot compute regression: all X values are identical (zero variance)."}), 400

    slope = ((n*sum_xy) - (sum_x*sum_y)) / denominator

    intercept = (sum_y - slope * sum_x) / n


    y_pred = []
    for i in range(n):
        y_pred.append(slope*x[i] + intercept)

    error = []
    for i in range(n):
        error.append(abs(y[i]-y_pred[i]))

    sum_error = sum(error)

    mae = sum_error/n

    error_square = []
    for i in range(n):
        error_square.append(error[i]**2)

    sum_error_square = sum(error_square)

    mse = sum_error_square/n

    rmse = mse ** 0.5

    yy_pred = [(i-j)**2 for i,j in zip(y,y_pred)  ]
    yy_mean = [(i-(sum(y)/len(y)))**2 for i in y]

    sum_yy_mean = sum(yy_mean)
    if sum_yy_mean == 0:
        r2 = 1.0
    else:
        r2 = 1.0 - (sum(yy_pred)/sum_yy_mean)


    # Convert to NumPy arrays
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y)
    predictions = np.array(y_pred)

    plt.figure(figsize=(6,4))

    plt.scatter(X, Y, color="blue", label="Actual Data")

    plt.plot(X, predictions,
             color="red",
             linewidth=2,
             label="Best Fit Line")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")

    plt.legend()
    
    # Save regression plot to a memory buffer and convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    regression_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Generate metrics scatter plot
    plt.figure(figsize=(6,4))
    metrics = ["MSE", "MAE", "RMSE", "R2"]
    metric_values = [mse, mae, rmse, r2]
    
    plt.scatter(metrics, metric_values, color="purple", s=100, marker="o", label="Metric Value")
    for i, val in enumerate(metric_values):
        plt.annotate(f"{val:.4f}", (metrics[i], metric_values[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
        
    plt.xlabel("Evaluation Metrics")
    plt.ylabel("Values")
    plt.title("Model Metrics Comparison")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save metrics plot to a memory buffer and convert to base64
    buf_metrics = io.BytesIO()
    plt.savefig(buf_metrics, format="png")
    buf_metrics.seek(0)
    metrics_base64 = base64.b64encode(buf_metrics.read()).decode("utf-8")
    plt.close()

    return jsonify({

        "equation": f"Y = {slope:.2f}x + {intercept:.2f}",

        "slope": round(float(slope), 4),

        "intercept": round(float(intercept), 4),

        "mae": round(float(mae), 4),

        "mse": round(float(mse), 4),

        "rmse": round(float(rmse), 4),

        "r2": round(float(r2), 4),

        "regression_image": f"data:image/png;base64,{regression_base64}",
        "metrics_image": f"data:image/png;base64,{metrics_base64}"

    })

if __name__ =="__main__":
    app.run(debug=True)
