from loguru import logger
import sys
import toml
from pathlib import Path

logger.remove()
base_path = (Path(__file__).parent / "../").resolve()
env_settings = toml.load(f"{base_path}/sidekick/configs/env.toml")
logger.add(sys.stderr, level=env_settings["LOGGING"]["LOG-LEVEL"])
