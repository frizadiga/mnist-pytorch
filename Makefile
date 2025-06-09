# Variables
PYTHON = python3
PIP = pip3
VENV_NAME = venv
REQUIREMENTS = requirements.txt
MODEL_FILE = mnist_model.pth
DATA_DIR = data
RESULTS_DIR = results
MAIN_SCRIPT = mnist_classifier.py
INFERENCE_SCRIPT = inference.py

# Default target
.PHONY: help
help:
	@echo "MNIST PyTorch Project Makefile"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Set up virtual environment and install dependencies"
	@echo "  install        - Install dependencies"
	@echo "  clean          - Clean generated files and cache"
	@echo "  clean-all      - Clean everything including data and models"
	@echo "  train          - Train the model (interactive)"
	@echo "  train-cnn      - Train CNN model"
	@echo "  train-fc       - Train fully connected model"
	@echo "  inference      - Run inference on custom image"
	@echo "  test           - Run quick test to verify installation"
	@echo "  download-data  - Download MNIST dataset"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black"
	@echo "  requirements   - Generate/update requirements.txt"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  benchmark      - Run training benchmark"
	@echo "  gpu-check      - Check GPU availability"
	@echo ""

# Setup virtual environment
.PHONY: setup
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Activating virtual environment and installing dependencies..."
	. $(VENV_NAME)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV_NAME)/bin/activate && $(PIP) install -r $(REQUIREMENTS)
	@echo "Setup complete! Activate with: source $(VENV_NAME)/bin/activate"

# Install dependencies
.PHONY: install
install:
	@echo "Installing dependencies..."
	$(PIP) install -r $(REQUIREMENTS)
	@echo "Dependencies installed!"

# Clean generated files
.PHONY: clean
clean:
	@echo "Cleaning generated files..."
	rm -f *.png
	rm -f $(MODEL_FILE)
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf .pytest_cache
	rm -rf $(RESULTS_DIR)
	@echo "Clean complete!"

# Clean everything including data
.PHONY: clean-all
clean-all: clean
	@echo "Cleaning all files including data..."
	rm -rf $(DATA_DIR)
	rm -rf $(VENV_NAME)
	@echo "Deep clean complete!"

# Train model (interactive)
.PHONY: train
train:
	@echo "Starting interactive training..."
	$(PYTHON) $(MAIN_SCRIPT)

# Train CNN model
.PHONY: train-cnn
train-cnn:
	@echo "Training CNN model..."
	echo "1" | $(PYTHON) $(MAIN_SCRIPT)

# Train FC model
.PHONY: train-fc
train-fc:
	@echo "Training Fully Connected model..."
	echo "2" | $(PYTHON) $(MAIN_SCRIPT)

# Run inference
.PHONY: inference
inference:
	@echo "Running inference..."
	$(PYTHON) $(INFERENCE_SCRIPT)

# Quick test
.PHONY: test
test:
	@echo "Running quick test..."
	$(PYTHON) -c "import torch; import torchvision; print('PyTorch version:', torch.__version__); print('Torchvision version:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"

# Download MNIST data
.PHONY: download-data
download-data:
	@echo "Downloading MNIST dataset..."
	$(PYTHON) -c "import torchvision; torchvision.datasets.MNIST('./data', train=True, download=True); torchvision.datasets.MNIST('./data', train=False, download=True)"
	@echo "MNIST dataset downloaded!"

# Code linting
.PHONY: lint
lint:
	@echo "Running linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 *.py; \
	else \
		echo "flake8 not installed. Install with: pip install flake8"; \
	fi

# Format code
.PHONY: format
format:
	@echo "Formatting code..."
	@if command -v black >/dev/null 2>&1; then \
		black *.py; \
	else \
		echo "black not installed. Install with: pip install black"; \
	fi

# Generate requirements
.PHONY: requirements
requirements:
	@echo "Generating requirements.txt..."
	$(PIP) freeze > $(REQUIREMENTS)
	@echo "Requirements updated!"

# Docker targets
.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	docker build -t mnist-pytorch .

.PHONY: docker-run
docker-run:
	@echo "Running Docker container..."
	docker run -it --rm -v $(PWD):/workspace mnist-pytorch

# Benchmark training
.PHONY: benchmark
benchmark:
	@echo "Running training benchmark..."
	@mkdir -p $(RESULTS_DIR)
	@echo "Benchmarking CNN..." && \
	(time echo "1" | $(PYTHON) $(MAIN_SCRIPT)) 2>&1 | tee $(RESULTS_DIR)/cnn_benchmark.log
	@echo "Benchmarking FC..." && \
	(time echo "2" | $(PYTHON) $(MAIN_SCRIPT)) 2>&1 | tee $(RESULTS_DIR)/fc_benchmark.log
	@echo "Benchmark results saved to $(RESULTS_DIR)/"

# Check GPU
.PHONY: gpu-check
gpu-check:
	@echo "Checking GPU availability..."
	$(PYTHON) -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

# Development dependencies
.PHONY: install-dev
install-dev:
	@echo "Installing development dependencies..."
	$(PIP) install black flake8 pytest jupyter notebook
	@echo "Development dependencies installed!"

# Run Jupyter notebook
.PHONY: notebook
notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook

# Create results directory
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

# Check if model exists
.PHONY: check-model
check-model:
	@if [ -f $(MODEL_FILE) ]; then \
		echo "Model file $(MODEL_FILE) exists"; \
		ls -lh $(MODEL_FILE); \
	else \
		echo "Model file $(MODEL_FILE) not found. Run 'make train' first."; \
	fi

# Show project status
.PHONY: status
status:
	@echo "Project Status"
	@echo "=============="
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Working directory: $(PWD)"
	@echo "Virtual environment: $(if $(VIRTUAL_ENV),Active ($(VIRTUAL_ENV)),Not active)"
	@echo ""
	@echo "Files:"
	@ls -la *.py 2>/dev/null || echo "No Python files found"
	@echo ""
	@echo "Model file:"
	@ls -lh $(MODEL_FILE) 2>/dev/null || echo "Model file not found"
	@echo ""
	@echo "Data directory:"
	@ls -la $(DATA_DIR) 2>/dev/null || echo "Data directory not found"
	@echo ""
	@echo "Dependencies:"
	@$(PIP) list | grep -E "(torch|numpy|matplotlib)" 2>/dev/null || echo "Core dependencies not found"

# Quick start (setup + train)
.PHONY: quickstart
quickstart: setup download-data train-cnn
	@echo "Quick start complete! CNN model trained and ready."

# Production targets
.PHONY: prod-setup
prod-setup:
	@echo "Setting up for production..."
	$(PIP) install --no-dev -r $(REQUIREMENTS)
	@echo "Production setup complete!"

# Package project
.PHONY: package
package: clean
	@echo "Packaging project..."
	@mkdir -p dist
	tar -czf dist/mnist-pytorch-$(shell date +%Y%m%d).tar.gz \
		--exclude=dist \
		--exclude=$(VENV_NAME) \
		--exclude=__pycache__ \
		--exclude=.git \
		.
	@echo "Project packaged in dist/"

# Show make targets
.PHONY: targets
targets:
	@echo "Available make targets:"
	@$(MAKE) -qp | awk -F':' '/^[a-zA-Z0-9][^$$#\/\t=]*:([^=]|$$)/ {split($$1,A,/ /);for(i in A)print A[i]}' | sort | uniq
