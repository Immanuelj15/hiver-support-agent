.PHONY: setup prepare-data index baseline demo test evaluate clean

setup:
	pip install -r requirements.txt

prepare-data:
	python scripts/prepare_data.py

index:
	python scripts/build_index.py

baseline:
	python -m baselines.tfidf_logistic

demo:
	python scripts/run_demo.py --interactive

test:
	pytest tests/ -v

evaluate:
	python -m src.evaluation.run_all

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
