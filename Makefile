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
	@echo "  make table3        - Run Table 3 experiments (all examples, congestion=0.00)"
	@echo "  make table4        - Run Table 4 experiments (selected examples, multiple congestions)"
	@echo "  make true_error    - Run a special example to compare with the exact transportation"
	@echo ""
	@echo "Environment Variables"
	@echo "------------------------------------"
	@echo "  tol=<tolerance>    - Set the tolerance for Table3 or Table4 experiments (default: $(tol))"
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

# Common params
PARAM = --ntime=31 --nit=10000 --time_limit=5000 --tol=$(tol)\
		--save --outdir=$(OUTDIR_MAIN)

EXTRA_HILLS = --power_perceptual=0.5

table3: EXAMPLES = ring refined_punctured_ball hand refined_hand refined_bunny refined_airplane refined_armadillo hills knots_3 knots_5
table3: CONGESTIONs = 0.00
table3: check-python $(OUTDIR_MAIN)
	@$(MAKE) run_experiments EXAMPLES="$(EXAMPLES)" CONGESTIONs="$(CONGESTIONs)"

table4: EXAMPLES = hills knots_3 knots_5
table4: CONGESTIONs = 0.00 0.01 0.05
table4: check-python $(OUTDIR_MAIN)
	@$(MAKE) run_experiments EXAMPLES="$(EXAMPLES)" CONGESTIONs="$(CONGESTIONs)"

run_experiments:
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
all: table3 table4 true_error
.PHONY: table3 table4 run_experiments true_error
