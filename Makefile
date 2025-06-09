# Basic
ifeq ($(OS),Windows_NT)
	PYTHON := .venv/Scripts/python.exe
else
	PYTHON := .venv/bin/python.exe
endif
export PYTHONPATH=./
export PYTHONIOENCODING=utf-8

ifeq ($(OS), Windows_NT)
    OUTDIR := output/$(shell powershell -Command "Get-Date -Format 'yyyy_MMdd_HHmm'")
else
    OUTDIR := output/$(shell date +%Y_%m%d_%H%M)
endif

$(OUTDIR):
	mkdir -p "$(OUTDIR)"

# =======================================
# Utils
# =======================================

# Default target
.DEFAULT_GOAL := help

help: check-python
	@echo "======= Welcome to DOTs-SOCP ======="
	@echo ""
	@echo "Benchmark"
	@echo "------------------------------------"
	@echo "  make main        	- Run the main comparison experiments corresponding to the tables"
	@echo "  make true_error  	- Run a special example to compare with the exact transportation"
	@echo ""
	@echo "Environment Variables"
	@echo "------------------------------------"
	@echo "  tol=<tolerance> 	- Set the tolerance for the main experiments (default: $(tol))"
	@echo ""

check-python:
	@if [ ! -f "$(PYTHON)" ]; then \
		echo "Python interpreter not found! Please check the configuration of PYTHON variable in Makefile" && exit 1; \
	fi

.PHONY: help check-python

# =======================================
# Comparison table of numerical experiments
# =======================================
OUTDIR_MAIN := $(OUTDIR)_main
tol ?= 1e-4

$(OUTDIR_MAIN):
	mkdir -p "$(OUTDIR_MAIN)"

# List of comparison experiments
EXAMPLES = airplane 		refined_airplane \
		   armadillo 		refined_armadillo \
		   hand 			refined_hand \
		   punctured_ball 	refined_punctured_ball \
		   bunny 			refined_bunny \
		   ring knots_3 knots_5 hills
CONGESTIONs = 0.00 0.01 0.05

# Common params
PARAM = --ntime=31 --nit=10000 --time_limit=5000 --tol=$(tol)\
		--save --outdir=$(OUTDIR_MAIN)

# Extra params for `hills` example
EXTRA_HILLS = --power_perceptual=0.5

main: check-python $(OUTDIR_MAIN)
	@for c_value in $(CONGESTIONs); do \
		out_dir="$(OUTDIR_MAIN)/congestion_$${c_value//./\_}"; \
		mkdir -p "$${out_dir}"; \
		info_log_file="$${out_dir}/info.log"; \
		for example in $(EXAMPLES); do \
			_extra_params_hills=''; \
            if [ "$${example}" = "hills" ]; then \
                _extra_params_hills=$(EXTRA_HILLS); \
            fi; \
			echo "Running: example=$${example}, congestion=$${c_value}, $${_extra_params_hills}" >&2; \
			$(PYTHON) replication/main.py \
				$(PARAM) \
				--example=$${example} \
				--congestion=$${c_value} \
				--log_file=$${info_log_file} \
				--outdir=$${out_dir} \
				$${_extra_params_hills}; \
		done; \
		$(PYTHON) replication/log2table.py --input "$${info_log_file}" --output "$${out_dir}/comparison_table.tex" "$${out_dir}/comparison_table.html"; \
	done;

# =======================================
# Error versus exact transportation
# =======================================
OUTDIR_TRUE_ERROR := $(OUTDIR)_true_error

$(OUTDIR_TRUE_ERROR):
	mkdir -p "$(OUTDIR_TRUE_ERROR)"

true_error: check-python $(OUTDIR_TRUE_ERROR)
	@$(PYTHON) replication/main_versus_exact.py \
		--example=plane \
		--tol=0.00001 \
		--nit=20000 \
		--save \
		--outdir=$(OUTDIR_TRUE_ERROR) \
		--log_file="$(OUTDIR_TRUE_ERROR)/info.log";

# =======================================
all: main true_error
.PHONY: main true_error
