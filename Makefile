demo_data = s3cmd get --recursive --skip-existing s3://h2o-sql-sidekick/demo/sleepEDA/ ./examples/demo/

.PHONY: download_demo_data

all: download_demo_data

setup: download_demo_data ## Setup
	python3 -m venv .sidekickvenv
	./.sidekickvenv/bin/python3 -m pip install --upgrade pip
	./.sidekickvenv/bin/python3 -m pip install wheel
	./.sidekickvenv/bin/python3 -m pip install -r requirements.txt
	mkdir -p ./examples/demo/


download_demo_data:
	mkdir -p ./examples/demo/
	$(demo_data)

run:
	./.sidekickvenv/bin/python3 start.py

clean:
	rm -rf ./db
	rm  -rf .var

cloud_bundle:
	h2o bundle -L debug 2>&1 | tee -a h2o-bundle.log
