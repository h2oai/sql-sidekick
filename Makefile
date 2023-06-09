sentence_transformer = s3cmd get --recursive --skip-existing s3://h2o-model-gym/models/nlp/sentence_trasnsformer/all-MiniLM-L6-v2/ ./models/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2

.PHONY: download-models

all: download-models

setup: download_models ## Setup
	python3 -m venv .sidekickvenv
	./.sidekickvenv/bin/python3 -m pip install --upgrade pip
	./.sidekickvenv/bin/python3 -m pip install wheel
	./.sidekickvenv/bin/python3 -m pip install -r requirements.txt

download_models:
	mkdir -p ./models/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2
	$(sentence_transformer)
