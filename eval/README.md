Usage:
1. `python3 -m venv .sidekick_venv`
2. `source .sidekick_venv/bin/activate`
3. `pip install --force-reinstall sql_sidekick-x.x.x-py3-none-any.whl` (# replace x.x.x with the latest version number), https://github.com/h2oai/sql-sidekick/releases
4. `python eval/inference.py run-eval --help`
```
Options:
  -i, --input_data_path TEXT  Path to dataset in .csv format
  -t, --table_name TEXT       Table name related to the supplied dataset
  -e, --eval_data_path TEXT   Path to eval dataset in .csv format
  -m, --model_name TEXT       Model name to use for inference
  -s, --sample_qna_path TEXT  Path to sample QnA in .csv format
  -n, --iterations INTEGER    Number of iterations to run
  -th, --threshold FLOAT      Similarity threshold
  -k, --kwargs TEXT           Additional arguments
```
5. python eval/inference.py run-eval -i <input_data.csv> -t "your_table_name" -e <eval_ground_truth.csv> -s <sample_qna.csv> -m "h2ogpt-sql-sqlcoder-34b-alpha"


Benchmarks: WIP
