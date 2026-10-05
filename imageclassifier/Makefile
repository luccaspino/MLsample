#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = imageclassifier
PYTHON_VERSION = 3.12.13
PYTHON_INTERPRETER = uv run python
RUFF = uv run ruff

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(RUFF) format --check
	$(RUFF) check


## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Format source code with ruff
.PHONY: format
format:
	$(RUFF) check --fix
	$(RUFF) format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Train the CIFAR-10 classifier
.PHONY: train-cifar10
train-cifar10:
	$(PYTHON_INTERPRETER) -m ic.main train-cifar10


## Train the cardiovascular classifier
.PHONY: train-heart
train-heart:
	$(PYTHON_INTERPRETER) -m ic.main train-heart



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
