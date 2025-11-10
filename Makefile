# Default target: run all analyses
.PHONY: all
all: analyses

# -------------------------------
# Clean: delete intermediate files but keep code and raw data
.PHONY: clean
clean:
	@echo "Cleaning cached files..."
	rm data/complete_dfs/*
	rm data/weather/*

# -------------------------------
# Run analyses (without downloading raw data or making current predictions)
.PHONY: analyses
analyses:
	@echo "Running all analyses..."
	python scripts/model_training.py

# -------------------------------
# Make current predictions
.PHONY: predictions
predictions:
	@echo "Generating predictions..."
	python scripts/make_predictions.py

# -------------------------------
# Download raw data (delete existing raw data first)
.PHONY: rawdata
rawdata:
	@echo "Refreshing raw data..."
	rm data/weather/*
	python scripts/download_raw_data.py