import os
from pathlib import Path

import pytest
from sidekick.prompter import db_setup
from sidekick.schema_generator import generate_schema
from sidekick.utils import setup_dir

base_path = (Path(__file__).parent / "../").resolve()
base_path = "."
cache_path = f"{base_path}/var/lib/tmp"
setup_dir(base_path)

HOST_NAME = "localhost"
USER_NAME = "sqlite"
PASSWORD = "abc"
DB_NAME = "query_test"
PORT = "5432"

data_path = "./examples/demo/sleep_health_and_lifestyle_dataset.csv"
# Replace table_name as needed
table_name = "sleep_health_and_lifestyle"
_, table_info_path = generate_schema(data_path=data_path, output_path=f"{cache_path}/{table_name}_table_info.jsonl")
# Set DB and table to test
# Set add_sample=False if no need to add rows to the table (default: = True)
# Initialize DB
if Path(f"{base_path}/db/sqlite/{DB_NAME}.db").exists():
    os.remove(f"{base_path}/db/sqlite/{DB_NAME}.db")


# Currently testing sqlite setup
def test_db_setup():
    res, err = db_setup(
                    db_name=DB_NAME,
                    hostname=HOST_NAME,
                    user_name=USER_NAME,
                    password=PASSWORD,
                    port=PORT,
                    table_info_path=table_info_path,
                    table_samples_path=data_path,
                    table_name=table_name,
                    local_base_path=base_path
    )
    assert err is None
    assert res > 0
