venv:
	python -m venv env

dependencies: env
	. env/bin/activate
	pip install -r requirements.txt
	pip install pytest

test: env
	. env/bin/activate
	pytest
	