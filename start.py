import os
import shlex
import subprocess
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def setup_dir(base_path: str):
    dir_list = ["var/lib/tmp/data", "var/lib/tmp/jobs", "var/lib/tmp/.cache", "models", "db/sqlite"]
    for _dl in dir_list:
        p = Path(f"{base_path}/{_dl}")
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)


print(f"Download models...")
base_path = (Path(__file__).parent).resolve() if os.path.isdir("./.sidekickvenv/bin/") else "/meta_data"
setup_dir(base_path)

# Model 1:
print(f"Download model 1...")
snapshot_download(repo_id="NumbersStation/nsql-llama-2-7B", cache_dir=f"{base_path}/models/")
# Model 2:
print(f"Download model 2...")
snapshot_download(repo_id="defog/sqlcoder2", cache_dir=f"{base_path}/models/")

print(f"Download embedding model...")
snapshot_download(repo_id="BAAI/bge-base-en", cache_dir=f"{base_path}/models/sentence_transformers/")

print("Starting SQL-Sidekick.")
DAEMON_PATH = "./.sidekickvenv/bin/uvicorn" if os.path.isdir("./.sidekickvenv/bin/") else "/resources/venv/bin/uvicorn"

cmd = f"{DAEMON_PATH} ui.app:main"
subprocess.check_output(shlex.split(cmd))
