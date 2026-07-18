import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("landing.html")

@app.route("/linear-regression")
def linear_regression():
    return render_template("linear_regression.html")

@app.route("/mlr")
def mlr():
    return render_template("mlr.html")

@app.route("/predict", methods=["POST"])
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

    yy_pred = [(i-j)**2 for i,j in zip(y,y_pred)]
    yy_mean = [(i-(sum(y)/len(y)))**2 for i in y]

    sum_yy_mean = sum(yy_mean)
    if sum_yy_mean == 0:
        r2 = 1.0
    else:
        r2 = 1.0 - (sum(yy_pred)/sum_yy_mean)

    X = np.array(x).reshape(-1, 1)
    Y = np.array(y)
    predictions = np.array(y_pred)

    plt.figure(figsize=(6,4))
    plt.scatter(X, Y, color="#06b6d4", edgecolor="none", s=40, label="Actual Data", alpha=0.8)
    plt.plot(X, predictions, color="#f43f5e", linewidth=2.5, label="Best Fit Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    regression_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    plt.figure(figsize=(6,4))
    metrics = ["MSE", "MAE", "RMSE", "R2"]
    metric_values = [mse, mae, rmse, r2]
    colors = ["#a855f7", "#06b6d4", "#f43f5e", "#10b981"]
    
    bars = plt.bar(metrics, metric_values, color=colors, alpha=0.85, width=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    plt.xlabel("Evaluation Metrics")
    plt.ylabel("Values")
    plt.title("Model Metrics Comparison")
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    
    buf_metrics = io.BytesIO()
    plt.savefig(buf_metrics, format="png", dpi=100)
    buf_metrics.seek(0)
    metrics_base64 = base64.b64encode(buf_metrics.read()).decode("utf-8")
    plt.close()

    return jsonify({
        "equation": f"Y = {slope:.4f}x + {intercept:.4f}",
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "regression_image": f"data:image/png;base64,{regression_base64}",
        "metrics_image": f"data:image/png;base64,{metrics_base64}"
    })

@app.route("/predict-mlr", methods=["POST"])
def predict_mlr():
    try:
        data = request.get_json()
        x_lists = data.get('X', [])
        y_list = data.get('Y', [])

        X_cols = []
        for col in x_lists:
            X_cols.append([float(val) for val in col])
        Y = [float(val) for val in y_list]
    except (ValueError, TypeError, AttributeError):
        return jsonify({"error": "Inputs must be lists of numbers."}), 400

    n = len(Y)
    num_features = len(X_cols)

    if n < 3:
        return jsonify({"error": "Multiple Linear Regression requires at least 3 data points."}), 400
    if num_features < 1:
        return jsonify({"error": "At least 1 feature column (X) is required."}), 400

    for i, col in enumerate(X_cols):
        if len(col) != n:
            return jsonify({"error": f"Feature X{i+1} has length {len(col)}, but target Y has length {n}."}), 400

    try:
        Y_arr = np.array(Y)
        X_arr = np.column_stack(X_cols)
        
        # Check for zero variance or extreme collinearity
        if np.all(np.std(X_arr, axis=0) == 0):
            return jsonify({"error": "Cannot compute regression: features have zero variance."}), 400
            
        X_with_intercept = np.column_stack([np.ones(n), X_arr])
        
        # Least squares solver (robust using pseudo-inverse)
        beta = np.linalg.pinv(X_with_intercept) @ Y_arr
        intercept = float(beta[0])
        slopes = [float(b) for b in beta[1:]]
        
        y_pred = X_with_intercept @ beta
        
        mae = float(np.mean(np.abs(Y_arr - y_pred)))
        mse = float(np.mean((Y_arr - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        
        y_mean = float(np.mean(Y_arr))
        ss_tot = float(np.sum((Y_arr - y_mean) ** 2))
        ss_res = float(np.sum((Y_arr - y_pred) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0)
    except Exception as e:
        return jsonify({"error": f"Calculation error: {str(e)}"}), 500

    # Build equation string
    eq_parts = [f"{intercept:.4f}"]
    for idx, slope in enumerate(slopes):
        sign = "+" if slope >= 0 else "-"
        eq_parts.append(f"{sign} {abs(slope):.4f}*X{idx+1}")
    equation = "Y = " + " ".join(eq_parts)

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(6,4))
    plt.scatter(Y_arr, y_pred, color="#06b6d4", edgecolor="none", s=40, label="Data Points", alpha=0.8)
    min_val = min(float(np.min(Y_arr)), float(np.min(y_pred)))
    max_val = max(float(np.max(Y_arr)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], color="#f43f5e", linestyle="--", linewidth=2, label="Perfect Fit (Y = Y_pred)")
    plt.xlabel("Actual Y")
    plt.ylabel("Predicted Y")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    mlr_plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Plot 2: Metrics Bar Chart
    plt.figure(figsize=(6,4))
    metrics_names = ["MSE", "MAE", "RMSE", "R2"]
    metrics_vals = [mse, mae, rmse, r2]
    colors = ["#a855f7", "#06b6d4", "#f43f5e", "#10b981"]
    bars = plt.bar(metrics_names, metrics_vals, color=colors, alpha=0.85, width=0.5)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.4f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
    plt.ylabel("Value")
    plt.title("Model Metrics Comparison")
    plt.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()

    buf_metrics = io.BytesIO()
    plt.savefig(buf_metrics, format="png", dpi=100)
    buf_metrics.seek(0)
    metrics_base64 = base64.b64encode(buf_metrics.read()).decode("utf-8")
    plt.close()

    coef_list = [{"name": "Intercept", "value": round(intercept, 4)}]
    for idx, slope in enumerate(slopes):
        coef_list.append({"name": f"Slope (X{idx+1})", "value": round(slope, 4)})

    return jsonify({
        "equation": equation,
        "coefficients": coef_list,
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "mlr_image": f"data:image/png;base64,{mlr_plot_base64}",
        "metrics_image": f"data:image/png;base64,{metrics_base64}"
    })

if __name__ == "__main__":
    app.run(debug=True)

