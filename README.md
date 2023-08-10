# sql-sidekick
A simple SQL assistant (WIP)


# Installation
## Dev
```
1. git clone git@github.com:h2oai/sql-sidekick.git
2. cd sql-sidekick
3. make setup
4. source ./.sidekickvenv/bin/activate
5. python sidekick/prompter.py
```
## Usage
```
Dialect: postgres
- docker pull postgres (will pull the latest version)
- docker run --rm --name pgsql-dev -e POSTGRES_PASSWORD=abc -p 5432:5432 postgres

Default: sqlite
Step: 
- Download and install .whl --> s3://sql-sidekick/releases/sql_sidekick-0.0.3-py3-none-any.whl
- python3 -m venv .sidekickvenv
- source .sidekickvenv/bin/activate
- python3 -m pip install sql_sidekick-0.0.3-py3-none-any.whl
```
## Start
```
Welcome to the SQL Sidekick! I am an AI assistant that helps you with SQL
queries. I can help you with the following:

  1. Configure a local database(for schema validation and syntax checking):
  `sql-sidekick configure db-setup -t "<local_dir_path_to_>/table_info.jsonl"` (e.g., format --> https://github.com/h2oai/sql-sidekick/blob/main/examples/telemetry/table_info.jsonl)

  2. Ask a question: `sql-sidekick query -q "avg Gpus" -s "<local_dir_path_to_>/samples.csv"` (e.g., format --> https://github.com/h2oai/sql-sidekick/blob/main/examples/telemetry/samples.csv)

  3. Learn contextual query/answer pairs: `sql-sidekick learn add-samples` (optional)

  4. Add context as key/value pairs: `sql-sidekick learn update-context` (optional)

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  configure  Helps in configuring local database.
  learn      Helps in learning and building memory.
  query      Asks question and returns SQL
```
