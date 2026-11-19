.PHONY: install test demo
install:
	pip install -r requirements.txt
test:
	python -m pytest -q
demo:
	PYTHONPATH=src python -m affordability.cli --demo
