setup: ## Setup
	mkdir -p ./var/lib/tmp/data
	mkdir -p ./var/lib/tmp/.cache
	python3 -m venv .sidekickvenv
	./.sidekickvenv/bin/python3 -m pip install --upgrade pip
	./.sidekickvenv/bin/python3 -m pip install wheel
	./.sidekickvenv/bin/python3 -m pip install -r requirements.txt
