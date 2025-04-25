.PHONY: all
all: setup

# Target to install dependencies using package-lock.json
.PHONY: install-dependencies
install-dependencies:
	npm install --legacy-peer-deps react-syntax-highlighter
	npm config set legacy-peer-deps true
	npm ci

# Setup target to run all necessary commands
.PHONY: setup
setup: install-dependencies
