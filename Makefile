LIB_NAME = shgp
TESTS_NAME = tests
NOTEBOOKS_NAME = docs/**/*.py
LINT_NAMES = src/$(LIB_NAME) $(TESTS_NAME)
TYPE_NAMES = src/$(LIB_NAME)
SUCCESS='\033[0;32m'
UNAME_S = $(shell uname -s)


install:
	pip install -d '.[dev]'

format: ## Formats code with `black` and `isort`
	@echo "\n=== Autoflake =============================================="
	autoflake --remove-all-unused-imports --recursive \
			--remove-unused-variables --in-place --exclude=__init__.py \
			$(LINT_NAMES)
	@echo "\n=== black =============================================="
	black $(LINT_NAMES)
	@echo "\n=== isort =============================================="
	isort $(LINT_NAMES)

check: ## Runs all static checks such as code formatting checks, linting, mypy
	@echo "\n=== flake8 (linting)===================================="
	flake8 --statistics
	@echo "\n=== black (formatting) ================================="
	black --check --diff $(LINT_NAMES)
	@echo "\n=== isort (formatting) ================================="
	isort --check --diff $(LINT_NAMES)
	@echo "\n=== mypy (static type checking) ========================"
	mypy $(TYPE_NAMES)
	mypy $(NOTEBOOKS_NAME)
