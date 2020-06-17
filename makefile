envname = env

venv:
	python3 -m venv $(envname)

dependencies: venv
	. $(envname)/bin/activate && \
	pip install -r requirements.txt

test: dependencies
	. $(envname)/bin/activate && \
	pytest
	
clean:
	rm -rf $(envname) __pycache__ .pytest_cache
