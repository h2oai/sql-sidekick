sentence_transformer = s3cmd get --recursive --skip-existing s3://h2o-model-gym/models/nlp/sentence_trasnsformer/all-MiniLM-L6-v1/ ./models/sentence_transformer/all-MiniLM-L6-v1

.PHONY: download-models

all: download-models

setup: download_models ## Setup
	mkdir -p ./var/lib/tmp/data
	mkdir -p ./var/lib/tmp/.cache
	python3 -m venv .sidekickvenv
	./.sidekickvenv/bin/python3 -m pip install --upgrade pip
	./.sidekickvenv/bin/python3 -m pip install wheel
	./.sidekickvenv/bin/python3 -m pip install -r requirements.txt

download_models:
	mkdir -p ./models/sentence_transformer/all-MiniLM-L6-v1
	$(sentence_transformer)
