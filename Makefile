.PHONY: install run run-real check-updates test lint app docker docker-up clean

install:            ## install package + dev tools
	pip install -e ".[dev]"

run:                ## full pipeline on synthetic data (no downloads)
	python -m geoai_prospector.pipeline --source synthetic

run-real:           ## full pipeline on public NT / GA data (see docs/DATA.md)
	python -m geoai_prospector.pipeline --source real

check-updates:      ## scheduled mode: only re-run if upstream data changed
	python -m geoai_prospector.pipeline --source real --if-changed

test:
	pytest -q

lint:
	ruff check src app tests

app:
	streamlit run app/streamlit_app.py

docker:
	docker build -t geoai-prospector .

docker-up:
	docker compose --profile pipeline run --rm pipeline && docker compose up -d app

clean:
	rm -rf data/interim data/processed outputs/* data/raw/synthetic
