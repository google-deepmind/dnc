
all: install run

install: venv
	: # Activate venv and install smthing inside
	mkdir tmp
	. venv/bin/activate && TMPDIR=tmp pip install -r requirements.txt
	rm -r tmp/

venv:
	: # Create venv if it doesn't exist
	: # test -d venv || virtualenv -p python3 --no-site-packages venv
	test -d venv || python -m venv venv

test: venv
	python -m pytest
	black .
	black dnc/
	black tests/

run:
	: # Run your app here, e.g
	: # determine if we are in venv,
	: # see https://stackoverflow.com/q/1871549
	bash -c ". venv/bin/activate && pip -V"

clean:
	rm -rf venv
	find -iname "*.pyc" -delete
	rm -rf logs
	rm -rf .pytest_cache
	rm -rf tmp/

