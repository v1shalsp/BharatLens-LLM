# === Imports ===
import os
import warnings
warnings.filterwarnings("ignore")

from apscheduler.schedulers.blocking import BlockingScheduler
from database import init_db, SessionLocal, MonitoredTopic
from analysis_logic import run_full_analysis_for_topic
from anomaly_detector import check_narrative_divergence
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Utility Functions ===
def setup_initial_topics():
    """Add default topics to the database if not already present."""
    print("[Progress] Setting up initial topics in the database...")
    with SessionLocal() as db:
        topics_to_monitor = {
            "Climate Change": "#climatechange OR #globalwarming",
            "AI Development": "#AI OR #ArtificialIntelligence",
            "Global Economy": "#economy OR #inflation"
        }
        for name, keywords in topics_to_monitor.items():
            if not db.query(MonitoredTopic).filter_by(name=name).first():
                db.add(MonitoredTopic(name=name, keywords=keywords))
        db.commit()
    print("[Progress] Initial topics setup complete.")

def trigger_analysis_for_all_topics():
    """Run analysis for all topics in parallel using ThreadPoolExecutor."""
    print("\n[Progress] --- Starting Scheduled Analysis Run ---")
    with SessionLocal() as db:
        topics = db.query(MonitoredTopic).all()
        with ThreadPoolExecutor(max_workers=8) as executor:  # Adjusted for 8-core CPU
            future_to_topic = {executor.submit(run_full_analysis_for_topic, topic): topic for topic in topics}
            for future in as_completed(future_to_topic):
                topic = future_to_topic[future]
                try:
                    future.result()
                    print(f"[Progress] Analysis complete for topic: {topic.name}")
                except Exception as e:
                    print(f"[Error] An error occurred during analysis for '{topic.name}': {e}")
    print("[Progress] --- Scheduled Analysis Run Finished ---\n")

# === Main Entrypoint ===
if __name__ == "__main__":
    print("[Progress] Initializing database...")
    init_db()
    print("[Progress] Database initialized.")
    setup_initial_topics()

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # Schedule the main analysis job (e.g., every hour)
    scheduler.add_job(trigger_analysis_for_all_topics, 'interval', hours=1, id='main_analysis_job')

    # Schedule the anomaly detection job (e.g., every 15 minutes)
    scheduler.add_job(check_narrative_divergence, 'interval', minutes=15, id='anomaly_detection_job')

    print("[Progress] Running initial tasks immediately on startup...")
    trigger_analysis_for_all_topics()
    print("[Progress] Running anomaly detection on startup...")
    check_narrative_divergence()

    print("[Progress] Autonomous analysis system starting... (Press Ctrl+C to exit)")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("[Progress] Scheduler stopped.")