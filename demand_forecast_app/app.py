from flask import Flask, request, render_template
import joblib
import numpy as np
import google.generativeai as genai
from flask_cors import CORS
import webbrowser
from threading import Timer
from datetime import datetime

app = Flask(__name__, template_folder="templates")
CORS(app)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


# -------------------------
# LOAD ML ELEMENTS
# -------------------------
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")
encoders = joblib.load("encoders.pkl")

# Remove Seasonality if previously present
if "Seasonality" in model_columns:
    model_columns.remove("Seasonality")

# Add Date manually
columns = ["Date"] + model_columns

# Gemini API
genai.configure(api_key="GOOGLE_API_KEY_HERE")

# Prediction history
history = []


@app.route("/")
def home():
    return render_template("index.html", columns=columns, history=history)


@app.route("/predict", methods=["POST"])
def predict():

    form_data = request.form.to_dict()

    # -------------------------
    # DATE PROCESSING
    # -------------------------
    raw_date = form_data.get("Date", "Missing")

    try:
        d = datetime.strptime(raw_date, "%Y-%m-%d")
        form_data["Day"] = d.day
        form_data["Month"] = d.month
        form_data["Year"] = d.year
    except:
        form_data["Day"], form_data["Month"], form_data["Year"] = 0, 0, 0

    # Remove original date
    form_data.pop("Date", None)

    # -------------------------
    # BUILD ML INPUT ROW
    # -------------------------
    row = []

    for col in model_columns:
        value = form_data.get(col, 0)

        if col in encoders:  
            try:
                value = encoders[col].transform([value])[0]
            except:
                value = 0
        else:
            try:
                value = float(value)
            except:
                value = 0

        row.append(value)

    row = np.array(row).reshape(1, -1)
    prediction = float(model.predict(row)[0])

    # -------------------------
    # INVENTORY RISK
    # -------------------------
    current_inventory = float(form_data.get("Inventory Level", 0))

    PD = prediction
    CI = current_inventory

    if CI < PD:
        risk = "High Stockout Risk"
        risk_color = "red"
    elif PD <= CI < PD * 1.2:
        risk = "Medium Risk (Tight Stock)"
        risk_color = "orange"
    elif CI >= PD * 1.2 and CI <= PD * 2:
        risk = "Safe Stock Level"
        risk_color = "green"
    else:
        risk = "Overstock Warning"
        risk_color = "yellow"

    # -------------------------
    # SAVE HISTORY (NOW WITH REGION + RISK)
    # -------------------------
    history.append({
        "Date": raw_date,
        "Category": form_data.get("Category", "NA"),
        "Region": form_data.get("Region", "NA"),
        "Inventory_Risk": risk,
        "prediction": prediction
    })

    # Keep last 20
    if len(history) > 20:
        history.pop(0)

    # -------------------------
    # AI ANALYSIS
    # -------------------------
    gem_model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Forecasted Demand: {prediction}
    Inputs: {form_data}
    Inventory Risk Level: {risk}

    Provide EXACTLY 3 short bullet points:
    - Why the demand might be predicted like this
    - How inventory risk affects business
    - What action to optimize stock
    """

    ai_text = gem_model.generate_content(prompt).text

    return render_template(
        "index.html",
        columns=columns,
        ml_prediction=round(prediction, 2),
        ai_text=ai_text,
        history=history,
        risk=risk,
        risk_color=risk_color
    )


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(debug=True)
