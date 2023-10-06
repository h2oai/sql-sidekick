sentence_transformer = s3cmd get --recursive --skip-existing s3://h2o-model-gym/models/nlp/sentence_trasnsformer/all-MiniLM-L6-v2/ ./models/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2
demo_data = s3cmd get --recursive --skip-existing s3://h2o-sql-sidekick/demo/sleepEDA/ ./examples/demo/

.PHONY: download_demo_data

all: download_demo_data

setup: download_demo_data ## Setup
	python3 -m venv .sidekickvenv
	./.sidekickvenv/bin/python3 -m pip install --upgrade pip
	./.sidekickvenv/bin/python3 -m pip install wheel
	./.sidekickvenv/bin/python3 -m pip install -r requirements.txt
	mkdir -p ./db/sqlite
	mkdir -p ./examples/demo/

download_models:
	mkdir -p ./models/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2

download_demo_data:
	mkdir -p ./examples/demo/
	$(demo_data)

cloud_bundle:
	h2o bundle -L debug 2>&1 | tee -a h2o-bundle.log


setup-doc:  # Install documentation dependencies
	cd documentation && npm install

run-doc:  # Run the doc locally
	cd documentation && npm start

update-documentation-infrastructure:
	cd documentation && npm update @h2oai/makersaurus
	cd documentation && npm ls

build-doc-locally:  # Bundles your website into static files for production
	cd documentation && npm run build

serve-doc-locally:  # Serves the built website locally
	cd documentation && npm run serve
