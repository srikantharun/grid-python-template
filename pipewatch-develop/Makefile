#!/usr/bin/env make -f

# ==============================================================================
# Variables
# ==============================================================================

# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------

# Shell settings
SHELL := /bin/bash
.SHELLFLAGS := -ecuo pipefail

# Make settings
.DEFAULT_GOAL := help

# Package tools settings
PIP_EXTRA_ARGS := \
	--extra-index-url http://10.1.5.124/repository/pypi-axe/simple \
	--trusted-host 10.1.5.124

# ------------------------------------------------------------------------------
# Other
# ------------------------------------------------------------------------------

# The Python version to use
PYTHON := python3.10

# The width of the help text target column
HELP_WIDTH := 30

# ==============================================================================
# Targets
# ==============================================================================

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help.
	@grep -E '^[a-zA-Z\_\.<>\-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-$(HELP_WIDTH)s\033[0m %s\n", $$1, $$2}'

.PHONY: clean
clean:  ## Clean up.
	@echo "Cleaning up..."
	@rm -rf .venv

# ------------------------------------------------------------------------------
# Project
# ------------------------------------------------------------------------------

.PHONY: install.package-tools
install.package-tools:  ## Install package-tools.
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv .venv
	@echo "Installing pipx..."
	@. .venv/bin/activate && \
		pip install --upgrade pip >/dev/null && \
		pip install pipx >/dev/null
	@echo "Installing package-tools..."
	@. .venv/bin/activate && \
		pipx install package-tools --python "$$(readlink -f "$$(which $(PYTHON))")" --pip-args "$(PIP_EXTRA_ARGS)"
	@echo "Cleaning up..."
	@rm -rf .venv

.PHONY: upgrade.package-tools
upgrade.package-tools:  ## Upgrade package-tools.
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv .venv
	@echo "Installing pipx..."
	@. .venv/bin/activate && \
		pip install --upgrade pip >/dev/null && \
		pip install pipx >/dev/null
	@echo "Upgrading package-tools..."
	@. .venv/bin/activate && \
		pipx upgrade package-tools --pip-args "$(PIP_EXTRA_ARGS)"
	@echo "Cleaning up..."
	@rm -rf .venv

.PHONY: uninstall.package-tools
uninstall.package-tools:  ## Uninstall package-tools.
	@echo "Creating virtual environment..."
	@$(PYTHON) -m venv .venv
	@echo "Installing pipx..."
	@. .venv/bin/activate && \
		pip install --upgrade pip >/dev/null && \
		pip install pipx >/dev/null
	@echo "Uninstalling package-tools..."
	@pipx uninstall package-tools
	@echo "Cleaning up..."
	@rm -rf .venv
