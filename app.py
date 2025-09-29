from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from flaml.automl import AutoML
import numpy as np

# -------------------------
# STEP 1: Generate datasets
# -------------------------
iris = load_iris(as_frame=True)
iris_df = iris.frame
iris_df.to_csv("iris.csv", index=False)

diabetes = load_diabetes(as_frame=True)
diabetes_df = diabetes.frame
diabetes_df["target"] = diabetes.target
diabetes_df.to_csv("diabetes.csv", index=False)

# -------------------------
# STEP 2: Train AutoML Models with FLAML
# -------------------------

# Classification Model (Iris)
X_cls = iris_df.drop(columns=["target"])
y_cls = iris_df["target"]

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

automl_cls = AutoML()
automl_cls.fit(X_train, y_train, task="classification", time_budget=60)

y_pred_cls = automl_cls.predict(X_test)
cls_accuracy = accuracy_score(y_test, y_pred_cls)

# Regression Model (Diabetes)
X_reg = diabetes_df.drop(columns=["target"])
y_reg = diabetes_df["target"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

automl_reg = AutoML()
automl_reg.fit(X_train_r, y_train_r, task="regression", time_budget=60)

y_pred_reg = automl_reg.predict(X_test_r)
reg_mse = mean_squared_error(y_test_r, y_pred_reg)

print(f"✅ Classification model accuracy: {cls_accuracy:.2f}")
print(f"✅ Regression model MSE: {reg_mse:.2f}")

# -------------------------
# STEP 3: Flask App
# -------------------------
app = Flask(__name__)

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AutoML Project (FLAML)</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #f4f6f8;
            color: #333;
        }

        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
        }

        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 40px;
        }

        h2 {
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 10px;
        }

        form {
            background: #fff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-top: 10px;
            margin-bottom: 5px;
            font-weight: 600;
        }

        input[type="text"] {
            width: 100%;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #ccc;
            margin-bottom: 15px;
            font-size: 14px;
            transition: border-color 0.3s;
        }

        input[type="text"]:focus {
            border-color: #3498db;
            outline: none;
        }

        input[type="submit"], button {
            background: #3498db;
            color: #fff;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s, transform 0.2s;
        }

        input[type="submit"]:hover, button:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }

        hr {
            margin: 40px 0;
            border: none;
            border-top: 2px solid #ecf0f1;
        }

        .result {
            background: #ecf0f1;
            padding: 15px 20px;
            border-radius: 10px;
            margin-top: 15px;
            font-weight: 600;
            font-size: 16px;
        }

        .cls-result {
            background: #dff9fb;
            color: #22a6b3;
        }

        .reg-result {
            background: #f6e58d;
            color: #f0932b;
        }

    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AutoML Project with FLAML</h1>

        <h2>🌸 Classification: Iris Flower</h2>
        <p>Trained Model Accuracy: <b>{{ cls_acc }}</b></p>
        <form method="post" action="/predict_classification">
            <label>Sepal Length:</label>
            <input type="text" name="sepal_length">
            <label>Sepal Width:</label>
            <input type="text" name="sepal_width">
            <label>Petal Length:</label>
            <input type="text" name="petal_length">
            <label>Petal Width:</label>
            <input type="text" name="petal_width">
            <input type="submit" value="Predict Flower">
        </form>

        {% if cls_prediction is not none %}
            <div class="result cls-result">🌟 Predicted Flower: {{ cls_prediction }}</div>
        {% endif %}

        <hr>

        <h2>📊 Regression: Diabetes Progression</h2>
        <p>Trained Model MSE: <b>{{ reg_mse }}</b></p>
        <form method="POST" action="/predict_regression">
            <label>Age:</label><input type="text" name="age">
            <label>Sex:</label><input type="text" name="sex">
            <label>BMI:</label><input type="text" name="bmi">
            <label>BP:</label><input type="text" name="bp">
            <label>S1:</label><input type="text" name="s1">
            <label>S2:</label><input type="text" name="s2">
            <label>S3:</label><input type="text" name="s3">
            <label>S4:</label><input type="text" name="s4">
            <label>S5:</label><input type="text" name="s5">
            <label>S6:</label><input type="text" name="s6">
            <button type="submit">Predict Diabetes Progression</button>
        </form>

        {% if reg_prediction is not none %}
            <div class="result reg-result">📈 Predicted Progression Value: {{ reg_prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE,
                                  cls_acc=f"{cls_accuracy:.2f}",
                                  reg_mse=f"{reg_mse:.2f}",
                                  cls_prediction=None,
                                  reg_prediction=None)

@app.route("/predict_classification", methods=["POST"])
def predict_classification():
    try:
        features = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"]),
        ]
        df = pd.DataFrame([features], columns=X_cls.columns)
        pred = automl_cls.predict(df)[0]
        flower_name = iris.target_names[int(pred)]
    except Exception as e:
        flower_name = f"Error: {e}"

    return render_template_string(HTML_TEMPLATE,
                                  cls_acc=f"{cls_accuracy:.2f}",
                                  reg_mse=f"{reg_mse:.2f}",
                                  cls_prediction=flower_name,
                                  reg_prediction=None)


@app.route("/predict_regression", methods=["POST"])
def predict_regression():
    try:
        features = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["bmi"]),
            float(request.form["bp"]),
            float(request.form["s1"]),
            float(request.form["s2"]),
            float(request.form["s3"]),
            float(request.form["s4"]),
            float(request.form["s5"]),
            float(request.form["s6"]),
        ]
        import pandas as pd

        # Convert input to DataFrame with the same column names as training
        df = pd.DataFrame([features], columns=X_reg.columns)

        # Predict
        pred = automl_reg.predict(df)[0]
    except Exception as e:
        pred = f"Error: {e}"

    return render_template_string(
        HTML_TEMPLATE,
        cls_acc=f"{cls_accuracy:.2f}",
        reg_mse=f"{reg_mse:.2f}",
        cls_prediction=None,
        reg_prediction=pred,
    )


# -------------------------
# STEP 4: Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
