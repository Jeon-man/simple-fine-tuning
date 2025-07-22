import os

from dotenv import load_dotenv

load_dotenv()

BASE_MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"

HF_USER_ID = os.getenv("HF_USER_ID")
MODEL_REPO_NAME = os.getenv("MODEL_REPO_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = f"{HF_USER_ID}/{MODEL_REPO_NAME}"