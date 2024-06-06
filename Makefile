demo_data = wget -O ./examples/demo/demo_data.csv --no-clobber https://h2o-sql-sidekick-public.s3.amazonaws.com/demo/Sleep_health_and_lifestyle_dataset.csv

.PHONY: download_demo_data

all: download_demo_data

setup: download_demo_data
	python -m venv .sidekickvenv
	./.sidekickvenv/bin/python -m pip install --upgrade pip
	./.sidekickvenv/bin/python -m pip install wheel
	./.sidekickvenv/bin/python -m pip install -r requirements.txt
	mkdir -p ./examples/demo/

download_demo_data:
	mkdir -p ./examples/demo/
	$(demo_data)

run:
	./.sidekickvenv/bin/python start.py

clean:
	rm -rf ./db
	rm -rf ./var

cloud_bundle:
	h2o bundle -L debug 2>&1 | tee -a h2o-bundle.log
