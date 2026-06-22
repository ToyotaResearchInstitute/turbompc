# TurboMPC Makefile
# Usage:
#   make install      # Full install: venv + pip + CUDA kernels
#   make cuda         # Build CUDA FFI kernels only
#   make test         # Run default test suite
#   make test-extended  # Run default + extended tests
#   make check        # Verify installation

SHELL := /bin/bash
PYTHON ?= python3
ifneq (,$(wildcard .venv))
VENV ?= .venv
else ifneq (,$(wildcard venv))
VENV ?= venv
else
VENV ?= .venv
endif
# Resolve to an absolute path so VENV works whether it is the default
# in-repo .venv, a relative path, or an external venv passed on the
# command line — e.g.  make install VENV=/path/to/existing/venv
VENV_ABS := $(abspath $(VENV))
PIP := $(VENV_ABS)/bin/pip
PYTEST := $(VENV_ABS)/bin/python -m pytest
CMAKE_SRC := turbompc/solvers/csrc
BUILD_DIR := build/ffi
TORCH_VENV ?= venv-torch
TORCH_PIP := $(TORCH_VENV)/bin/pip
TORCH_PKG ?= torch==2.10.0

.PHONY: install install-cpu venv pip pip-cpu cuda test test-cpu test-extended check check-cpu torch-env clean help

help:
	@echo "TurboMPC — Differentiable MPC on GPU"
	@echo ""
	@echo "  make install      Full install (venv + pip + CUDA)"
	@echo "  make install-cpu  CPU-only install (no GPU required)"
	@echo "  make cuda         Build CUDA FFI kernels only"
	@echo "  make test         Run default test suite"
	@echo "  make test-cpu     Run CPU-only tests (no GPU required)"
	@echo "  make test-extended  Run default + extended tests"
	@echo "  make check        Verify installation"
	@echo "  make check-cpu    Verify CPU-only installation"
	@echo "  make torch-env    Create isolated Torch CUDA env (no JAX conflicts)"
	@echo "  make clean        Remove build artifacts"
	@echo "  make clean-all    Remove build artifacts and venv (full reset)"

# ── Full install ─────────────────────────────────────────────────
install: venv pip cuda check
	@echo ""
	@echo "✓ Installation complete. Activate with: source $(VENV_ABS)/bin/activate"

# ── CPU-only install (no GPU required) ───────────────────────────
install-cpu: venv pip-cpu check-cpu
	@echo ""
	@echo "✓ CPU installation complete. Activate with: source $(VENV_ABS)/bin/activate"

# ── Virtual environment ──────────────────────────────────────────
venv:
	@if [ ! -d "$(VENV_ABS)" ]; then \
		echo "Creating virtual environment at $(VENV_ABS)..."; \
		$(PYTHON) -m venv $(VENV_ABS); \
	fi

# ── Python dependencies ─────────────────────────────────────────
pip: venv
	@echo "Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -U "jax[cuda12]" nvidia-cusparse-cu12 "nvidia-cudss-cu12>=0.8"
	$(PIP) install -e ".[dev]"
	@echo "Patching activate script with CUDA library paths..."
	@CUDA_LIBS="$$(find $(VENV_ABS)/lib -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | tr '\n' ':')"; \
	if [ -n "$${CUDA_LIBS}" ] && ! grep -q 'turbompc CUDA libs' $(VENV_ABS)/bin/activate 2>/dev/null; then \
		printf '\n# turbompc CUDA libs (nvidia pip packages)\nexport LD_LIBRARY_PATH="%s$$LD_LIBRARY_PATH"\n' "$${CUDA_LIBS}" >> $(VENV_ABS)/bin/activate; \
	fi
	@echo "Verifying JAX GPU..."
	@CUDA_LIBS="$$(find $(VENV_ABS)/lib -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | tr '\n' ':')"; \
	export LD_LIBRARY_PATH="$${CUDA_LIBS}$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}"; \
	$(VENV_ABS)/bin/python -c "import jax; devs=jax.devices(); print(f'JAX devices: {devs}'); assert any(d.platform in (\"gpu\",\"cuda\") for d in devs), 'No GPU found!'"

# ── CPU-only Python dependencies ────────────────────────────────
pip-cpu: venv
	@echo "Installing CPU-only Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -U "jax[cpu]"
	$(PIP) install -e ".[dev]"
	@echo "Verifying JAX CPU..."
	$(VENV_ABS)/bin/python -c "import jax; devs=jax.devices(); print(f'JAX devices: {devs}')"

# ── CUDA FFI kernels ────────────────────────────────────────────
cuda: venv
	@echo "Building CUDA FFI kernels..."
	@if [ ! -f "$(BUILD_DIR)/CMakeCache.txt" ]; then \
		cmake -S $(CMAKE_SRC) -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
			-DPython3_EXECUTABLE=$(VENV_ABS)/bin/python \
			-DPython3_FIND_VIRTUALENV=ONLY; \
	fi
	@CUDA_LIBS="$$(find $(VENV_ABS)/lib -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | tr '\n' ':')"; \
	export LD_LIBRARY_PATH="$${CUDA_LIBS}$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}"; \
	cmake --build $(BUILD_DIR) -j

# ── Tests ────────────────────────────────────────────────────────
test: cuda
	$(PYTEST) tests/

test-cpu:
	$(PYTEST) tests/python/

test-extended: cuda
	$(PYTEST) --run-extended tests/

# ── Verification ─────────────────────────────────────────────────
check: cuda
	@CUDA_LIBS="$$(find $(VENV_ABS)/lib -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | tr '\n' ':')"; \
	export LD_LIBRARY_PATH="$${CUDA_LIBS}$${LD_LIBRARY_PATH:+:$${LD_LIBRARY_PATH}}"; \
	$(VENV_ABS)/bin/python check_install.py

check-cpu:
	$(VENV_ABS)/bin/python check_install.py

# ── Optional: separate Torch CUDA env ───────────────────────────
torch-env:
	@if [ ! -d "$(TORCH_VENV)" ]; then \
		echo "Creating Torch virtual environment..."; \
		$(PYTHON) -m venv $(TORCH_VENV); \
	fi
	@echo "Installing Torch into $(TORCH_VENV) (kept separate from $(VENV_ABS))..."
	$(TORCH_PIP) install --upgrade pip
	$(TORCH_PIP) install --index-url https://download.pytorch.org/whl/cu128 "$(TORCH_PKG)"
	@echo "Verifying Torch CUDA..."
	@$(TORCH_VENV)/bin/python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda version', torch.version.cuda)"
	@echo ""
	@echo "✓ Torch env ready. Activate with: source $(TORCH_VENV)/bin/activate"

# ── Cleanup ──────────────────────────────────────────────────────
clean:
	rm -rf $(BUILD_DIR) build/ffi_trace build/ffi_py310

clean-all: clean
	rm -rf $(VENV_ABS) *.egg-info
