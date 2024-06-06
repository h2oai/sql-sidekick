# sql-sidekick
A simple SQL assistant (WIP)
Turn ★ into ⭐ (top-right corner) if you like the project! 🙏

## Motivation
- Historically, it’s common for data to be stored in Databases, democratizing insight generation.
- Enable a helpful assistant to help write complex queries across different database dialects with acceptable efficient execution accuracy (not just matching accuracy)
- Push to derive consistent generation without errors using smaller OSS models to save on compute costs.
- Provide a toolkit for users to mix and match different model sizes to optimize compute cost - e.g., smaller models for generation, remote bigger models for syntax correction or spell correction …
- Build a smart search engine for Databases/structured data, Text to SQL as a Natural Language interface (NLI) for data analysis


## Key Features
- An interactive UI to capture feedback along with a python-client and CLI mode.
- Ability for auto DB schema generation for input data using custom input format.
- Support for in-context learning (ICL) pipeline with RAG support to control hallucination
- Guardrails: to check for SQL injections via SELECT statements, e.g., `SELECT * FROM SleepStudy WHERE user_id = 11 OR 1=1;`
- Entity mapping/Schema linking: Ability to build memory for mapping business context to the data schema dynamically; **Note: currently enabled only via CLI, others WIP.
- Ability to save the chat history of query/answer pairs for future reference and improvements.
- Self-correction loop back: Validates syntactic correction of generation. **Note: Self-correction is currently enabled for all openAI GPT models. WIP for other OSS models.
- Integration with different database dialects - currently, SQLite/Postgres(_might be broken temporarily_)/Databricks is enabled. WIP to add support for Duckdb and others.
- Debug mode: Ability to evaluate/modify and validate SQL query against the configured database via UI
- Recommend sample questions: Often, given a dataset, we are unsure what to ask. To come around this problem, we have enabled the ability to generate recommendations for possible questions.

# Installation

## Requirements
This project requires Python version to be within the range "3.8.1" to "3.10.0". You can check your Python version by running the following command in your terminal:

```
python --version
```
If your Python version is not within the specified range, you may need to update or downgrade it.

## Dev
```
1. git clone git@github.com:h2oai/sql-sidekick.git
2. cd sql-sidekick
3. make setup
4. source ./.sidekickvenv/bin/activate
5. poetry install (in case there is an error, try `poetry update` before `poetry install`)
6. python sidekick/prompter.py
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
`sql-sidekick`

Welcome to the SQL Sidekick! I am an AI assistant that helps you with SQL
queries. I can help you with the following:
  0. Generate input schema:
  `sql-sidekick configure generate_schema configure generate_schema --data_path "./sample_passenger_statisfaction.csv" --output_path "./table_config.jsonl"`

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

## UI
### Steps to start locally
1. Download wave serve [0.26.3](https://github.com/h2oai/wave/releases/tag/v0.26.3)
2. `tar -xzf wave-0.26.3-linux-amd64`; `./waved -max-request-size="20M"`
3. Download the latest bundle: https://github.com/h2oai/sql-sidekick/releases/latest
4. unzip `ai.h2o.wave.sql-sidekick.x.x.x.wave`
5. make setup
6. source ./.sidekickvenv/bin/activate
7. make run
<img width="1670" alt="Screen Shot 2023-11-15 at 6 19 14 PM" src="https://github.com/h2oai/sql-sidekick/assets/1318029/5cf8a3ef-0d36-4416-ae2f-52672024fead">

## Citation & Acknowledgment
Please consider citing our project if you find it useful:

## Blogs:
https://medium.com/the-story-within/state-of-text-to-sql-dc3e3e4f8c64

```bibtex
@software{sql-sidekick,
    title = {{sql-sidekick: A simple SQL assistant}},
    author = {Pramit Choudhary, Michal Malohlava, Narasimha Durgam, Robin Liu, h2o.ai Team}
    url = {https://github.com/h2oai/sql-sidekick},
    year = {2024}
}
```
LLM frameworks adopted: [h2ogpt](https://github.com/h2oai/h2ogpt), [h2ogpte](https://pypi.org/project/h2ogpte/), [LangChain](https://github.com/langchain-ai/langchain), [llama_index](https://github.com/run-llama/llama_index), [openai](https://openai.com/blog/openai-api)
