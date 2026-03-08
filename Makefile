PY=PYTHONPATH=src python

setup:
	python -m pip install -r requirements.txt

test:
	PYTHONPATH=src pytest -q

smoke:
	$(PY) scripts/smoke_test_pipeline.py
