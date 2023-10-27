import os
import shlex
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

print(f"Download model...")
base_path = (Path(__file__).parent).resolve()
# Model 1:
print("Downloading nsql-llama-2-7B model...")
snapshot_download(repo_id="NumbersStation/nsql-llama-2-7B", cache_dir=f"{base_path}/models/")

# Model 2:
print("Downloading sqlcoder2 model...")
snapshot_download(repo_id="defog/sqlcoder2", cache_dir=f"{base_path}/models/")

# Model 3:
print("Downloading AquilaSQL-7B model...")
snapshot_download(repo_id="BAAI/AquilaSQL-7B", cache_dir=f"{base_path}/models/")

snapshot_download(repo_id="BAAI/bge-base-en", cache_dir=f"{base_path}/models/sentence_transformers/")
print(f"Download embedding model...")

print("Starting SQL-Sidekick.")
DAEMON_PATH = "./.sidekickvenv/bin/uvicorn" if os.path.isdir("./.sidekickvenv/bin/") else "/resources/venv/bin/uvicorn"

cmd = f"{DAEMON_PATH} ui.app:main"
subprocess.check_output(shlex.split(cmd))
