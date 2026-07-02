from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE = os.path.dirname(__file__)

iso_model    = joblib.load(os.path.join(BASE, "iso_model.pkl"))
kmeans_model = joblib.load(os.path.join(BASE, "kmeans_model.pkl"))
scaler       = joblib.load(os.path.join(BASE, "scaler.pkl"))

N_FEATURES = scaler.n_features_in_  # reads it automatically, no hardcoding
print(f"Models loaded. Scaler expects {N_FEATURES} features.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)  # force=True handles content-type issues

        if not data or "features" not in data:
            return jsonify({"error": "No features received"}), 400

        row = np.zeros((1, N_FEATURES))
        for i, val in enumerate(data["features"].values()):
            if i < N_FEATURES:
                row[0, i] = float(val)

        row_scaled = scaler.transform(row)

        iso_raw   = iso_model.predict(row_scaled)[0]
        iso_label = "ATTACK" if iso_raw == -1 else "NORMAL"
        cluster   = int(kmeans_model.predict(row_scaled)[0])

        return jsonify({
            "isolation_forest": {"label": iso_label},
            "kmeans":           {"cluster": cluster},
            "overall":          iso_label
        })

    except Exception as e:
        # This now returns JSON instead of HTML crash page
        print(f"ERROR in /predict: {e}")  # prints to your terminal
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": N_FEATURES})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)