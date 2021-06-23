all: install

install: venv
	: # Activate venv and install requirements
	mkdir tmp
	source venv/bin/activate && TMPDIR=tmp pip install -r requirements.txt
	rm -r tmp/
	pre-commit install

venv:
	: # Create venv if it doesn't exist
	: # test -d venv || virtualenv -p python3 --no-site-packages venv
	test -d venv || python -m venv venv

test: venv
	source venv/bin/activate && python -m pytest

clean:
	rm -rf venv/
	find -iname "*.pyc" -delete
	rm -rf logs/
	rm -rf .pytest_cache
	rm -rf tmp/

