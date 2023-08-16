import logging
import subprocess
import shlex

logging.info("Starting SQL-Sidekick.")
DAEMON_PATH = "/resources/venv/bin/uvicorn"

cmd = f"{DAEMON_PATH} ui.app:main"
subprocess.check_output(shlex.split(cmd))
