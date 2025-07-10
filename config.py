# config.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# --- API Keys ---
# It's best practice to set these as environment variables.
# The code will look for the variable name (e.g., NEWS_API_KEY).
# If it's not found, it will use the placeholder string.

# Your key for https://newsapi.org/
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "b78e862e1bb74cdba1748cc5ef292880")

# Your Bearer Token for the X (Twitter) API v2
# https://developer.twitter.com/en/docs/authentication/oauth-2-0/bearer-tokens
# --- FIX: Hardcoded key removed for security ---
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "AAAAAAAAAAAAAAAAAAAAAHJj1AEAAAAAfNuvpIXo9Bnw8VWJNJFQdE8AW1U%3DXKAUwIHTzB0UofaikmPBrbA13wJKedlgbyVQr49fHhxNSR1VeW")

# --- File Paths ---
# Defines where to store project data and generated images.

# Get the absolute path of the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to store generated images for the frontend
VISUALS_PATH = os.path.join(BASE_DIR, 'static', 'visuals')

# Path to store persistent data like the temporal bias model
PROJECT_DATA_PATH = os.path.join(BASE_DIR, 'project_data')