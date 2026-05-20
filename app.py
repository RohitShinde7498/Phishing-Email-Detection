"""
Flask API — Phishing Email Detection System
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os, json, threading
from model import PhishingDetector, get_training_data, analyze_batch

app = Flask(__name__, template_folder="templates")
CORS(app)

# Global detector
detector = None
training_status = {"status": "idle", "progress": 0, "metrics": None}


def load_or_train():
    global detector, training_status
    training_status = {"status": "training", "progress": 10, "metrics": None}
    if os.path.exists("phishing_model.pkl"):
        try:
            detector = PhishingDetector.load("phishing_model.pkl")
            training_status = {"status": "ready", "progress": 100, "metrics": detector.metrics}
            print("[+] Loaded existing model")
            return
        except:
            pass
    # Train fresh
    detector = PhishingDetector()
    emails, labels = get_training_data()
    training_status["progress"] = 30
    metrics = detector.train(emails, labels)
    detector.save("phishing_model.pkl")
    training_status = {"status": "ready", "progress": 100, "metrics": metrics}
    print("[+] Model trained and saved")


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify(training_status)


@app.route("/api/metrics")
def api_metrics():
    if not detector:
        return jsonify({"error": "Model not ready"}), 503
    return jsonify({
        "metrics": detector.metrics,
        "feature_importances": detector.feature_importances,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not detector or training_status["status"] != "ready":
        return jsonify({"error": "Model not ready yet"}), 503
    body = request.get_json(force=True)
    email_text = (body.get("email") or "").strip()
    if not email_text:
        return jsonify({"error": "email field is required"}), 400
    try:
        result = detector.predict(email_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def api_batch():
    if not detector or training_status["status"] != "ready":
        return jsonify({"error": "Model not ready"}), 503
    body = request.get_json(force=True)
    emails = body.get("emails", [])
    if not emails:
        return jsonify({"error": "emails list is required"}), 400
    try:
        result = analyze_batch(detector, emails)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    global detector, training_status
    training_status = {"status": "training", "progress": 0, "metrics": None}
    t = threading.Thread(target=load_or_train, daemon=True)
    t.start()
    return jsonify({"message": "Retraining started"})


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    print("[*] Starting Phishing Detection Server...")
    t = threading.Thread(target=load_or_train, daemon=True)
    t.start()
    app.run(debug=False, host="0.0.0.0", port=5000)
