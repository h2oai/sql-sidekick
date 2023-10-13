import os
import shlex
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger as logging

logging.info(f"Download model...")
base_path = (Path(__file__).parent).resolve()
# Model 1:
snapshot_download(repo_id="NumbersStation/nsql-llama-2-7B", cache_dir=f"{base_path}/models/")
# Model 2:
snapshot_download(repo_id="defog/sqlcoder2", cache_dir=f"{base_path}/models/")
logging.info(f"Download embedding model...")
snapshot_download(repo_id="BAAI/bge-base-en", cache_dir=f"{base_path}/models/sentence_transformers/")

logging.info("Starting SQL-Sidekick.")
DAEMON_PATH = "./.sidekickvenv/bin/uvicorn" if os.path.isdir("./.sidekickvenv/bin/") else "/resources/venv/bin/uvicorn"

cmd = f"{DAEMON_PATH} ui.app:main"
subprocess.check_output(shlex.split(cmd))
