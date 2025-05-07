import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Summary, Gauge

# Tentukan path model relatif ke lokasi script ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "Workflow-CI", "MLProject", "artifacts", "model", "random_forest_model.pkl")

print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)

# definisi metrik
REQ_COUNT = Counter("api_requests_total", "Total API requests")
LATENCY   = Summary("inference_latency_seconds", "Inference latency")
LAST_PRED = Gauge("last_prediction", "Last predicted class (1/2/3)")

app = Flask(__name__)

@LATENCY.time()
@app.route("/predict", methods=["POST"])
def predict():
    REQ_COUNT.inc()
    try:
        data = pd.DataFrame([request.get_json()])
        pred = model.predict(data)[0]
        LAST_PRED.set({"Low":1,"Medium":2,"High":3}[pred])
        return jsonify(prediction=pred)
    except Exception as e:
        import traceback
        traceback.print_exc()   # cetak stacktrace ke console
        return jsonify(error=str(e)), 500
    
if __name__=="__main__":
    print("Starting Prometheus on :8000 and Flask on :5000")
    start_http_server(8000)
    app.run(host="0.0.0.0", port=5000)

