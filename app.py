# === Imports ===
import os
import logging
import math
import numpy as np
from flask import Flask, request, jsonify, abort, render_template, send_from_directory, send_file
from datetime import datetime, timedelta
from sqlalchemy import desc, func

# === Project Imports ===
import config
from database import SessionLocal, MonitoredTopic, AnalysisResult, init_db
from analysis_logic import run_on_demand_analysis

# === Optional Caching ===
try:
    from flask_caching import Cache
except ImportError:
    Cache = None
    print("[Error] Flask-Caching is not installed. Caching will not work. Run: pip install Flask-Caching")

# === Flask App Setup ===
app = Flask(__name__, static_folder='static', template_folder='templates')
init_db()

# === Configure Flask-Caching ===
if Cache is not None:
    cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 1800})  # 30 min cache
else:
    cache = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Utility Functions ===
def safe_convert(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def make_json_safe(d):
    if isinstance(d, dict):
        return {k: make_json_safe(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [make_json_safe(x) for x in d]
    else:
        return safe_convert(d)

def clean_json_numbers(obj):
    """Recursively replace NaN, Infinity, and -Infinity with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_json_numbers(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_numbers(x) for x in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

# === API Routes ===
@app.route('/')
def index():
    """Serve the main frontend page."""
    print("[DEBUG] Received request for / (index)")
    try:
        print("[DEBUG] Attempting to render index.html...")
        result = render_template('index.html')
        print("[DEBUG] Successfully rendered index.html.")
        return result
    except Exception as e:
        print(f"[ERROR] Exception while rendering index.html: {e}")
        import traceback
        traceback.print_exc()
        return "<h1>Internal Server Error</h1><pre>" + str(e) + "</pre>", 500

@app.route('/static/visuals/<path:filename>')
def serve_visuals(filename):
    """Serve visualization images."""
    print(f"[Progress] Serving static visual: {filename}")
    return send_from_directory(config.VISUALS_PATH, filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_topic_on_demand():
    """
    Receives a topic, triggers a real-time analysis, and returns the results as JSON.
    Handles caching, error handling, and ensures JSON serializability.
    """
    print("[Progress] Received /api/analyze POST request")
    data = request.get_json(silent=True)
    if not data or 'topic' not in data:
        print("[Error] Bad request: missing 'topic' in JSON body.")
        abort(400, description="Bad Request: JSON body must contain a 'topic' key.")

    topic_name = data['topic'].strip()
    if not topic_name:
        print("[Error] Bad request: topic is empty.")
        abort(400, description="Topic cannot be empty.")

    cache_key = f"analysis_{topic_name.lower()}"
    if cache is not None:
        cached_result = cache.get(cache_key)
        if cached_result:
            print(f"[Progress] Returning cached analysis for topic: {topic_name}")
            safe_results = make_json_safe(cached_result)
            cleaned_results = clean_json_numbers(safe_results)
            return jsonify(cleaned_results)

    try:
        print(f"[Progress] Triggering on-demand analysis for topic: {topic_name}")
        results_dict = run_on_demand_analysis(topic_name)
        print(f"[Progress] Analysis complete for topic: {topic_name}")
        if cache is not None:
            cache.set(cache_key, results_dict)
        safe_results = make_json_safe(results_dict)
        cleaned_results = clean_json_numbers(safe_results)
        return jsonify(cleaned_results)
    except ValueError as e: # Catches "No articles found" error from the logic
        print(f"[Error] {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"[Error] An unexpected error occurred during analysis for '{topic_name}': {e}")
        logging.error(f"An unexpected error occurred during analysis for '{topic_name}': {e}")
        return jsonify({"error": "An internal server error occurred during analysis."}), 500

# === Main Entrypoint ===
if __name__ == '__main__':
    print("[DEBUG] Flask app is starting...")
    print("[Progress] Ensuring visuals and project data directories exist...")
    os.makedirs(config.VISUALS_PATH, exist_ok=True)
    os.makedirs(config.PROJECT_DATA_PATH, exist_ok=True)
    print("[Progress] Starting Flask server on http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)