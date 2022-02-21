.ONESHELL:
ENV_PREFIX=$(shell python3 -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
USING_POETRY=$(shell grep "tool.poetry" pyproject.toml && echo "yes")

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

PHONY: clean
clean:            ## Clean unused files.
	-find ./ -name '*.pyc' -exec rm -f {} \;
	-find ./ -name '__pycache__' -exec rm -rf {} \;
	-find ./ -name 'Thumbs.db' -exec rm -f {} \;
	-find ./ -name '*~' -exec rm -f {} \;
	-rm -rf .cache
	-rm -rf .pytest_cache
	-rm -rf .mypy_cache
	-rm -rf build
	-rm -rf dist
	-rm -rf *.egg-info
	-rm -rf htmlcov
	-rm -rf .tox/
	-rm -rf docs/_build

.PHONY: virtualenv
virtualenv:       ## Create a virtual environment.
	@if [ "$(USING_POETRY)" ]; then poetry install && exit; fi
	@echo "creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@./.venv/bin/pip install -r requirements.txt
	@echo
	@echo "!!! Please run 'source .venv/bin/activate' to enable the environment !!!"

.PHONY: clean virtualenv test
test: virtualenv
	$(ENV_PREFIX)pip install kupy --upgrade --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/
	$(ENV_PREFIX)python __main__.py
