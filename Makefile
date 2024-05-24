# Define the Python interpreter
PYTHON = python3

# Define the R interpreter
RSCRIPT = Rscript

# Define your Python script
PYTHON_SCRIPT = python/model.py

# Define your R script
R_SCRIPT = analysis.R

# Default target
all: run_python run_r

# Run Python script with --train
train:
	$(PYTHON) $(PYTHON_SCRIPT) --train

# Run Python script without --train
notrain:
	$(PYTHON) $(PYTHON_SCRIPT)

# Run R script
run_r:
	$(RSCRIPT) $(R_SCRIPT)

# Run Python script
run_python: train notrain