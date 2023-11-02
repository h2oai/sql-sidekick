import os
import shlex
import subprocess
import time
from pathlib import Path

from huggingface_hub import snapshot_download

print(f"Download model...")
base_path = (Path(__file__).parent).resolve()

MODEL_CHOICE_MAP = {
    "h2ogpt-sql-sqlcoder2": "defog/sqlcoder2",
    "h2ogpt-sql-nsql-llama-2-7B": "NumbersStation/nsql-llama-2-7B",
}

for _m in MODEL_CHOICE_MAP.values():
    print(f"Downloading {_m}...", flush=True)
    snapshot_download(repo_id=_m, cache_dir=f"{base_path}/models/")
    time.sleep(3)

print(f"Download embedding model...")
snapshot_download(repo_id="BAAI/bge-base-en", cache_dir=f"{base_path}/models/sentence_transformers/")

print("Starting SQL-Sidekick.")
DAEMON_PATH = "./.sidekickvenv/bin/uvicorn" if os.path.isdir("./.sidekickvenv/bin/") else "/resources/venv/bin/uvicorn"

cmd = f"{DAEMON_PATH} ui.app:main"
subprocess.check_output(shlex.split(cmd))
