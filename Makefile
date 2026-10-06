PYTHON ?= python

.PHONY: test dry-run matrix plan data references paper
test:
	$(PYTHON) -m pytest -q
dry-run:
	$(PYTHON) -m train.main --config configs/M3.yaml --customers 20 --dry-run --set device=cpu training.batch_size=2
matrix:
	$(PYTHON) -m scripts.experiment_matrix
plan:
	$(PYTHON) -m scripts.run_experiments
data:
	$(PYTHON) -m scripts.generate_data --output datasets-regenerated
references:
	$(PYTHON) -m scripts.prepare_references --datasets datasets --time-limit 1
paper:
	$(PYTHON) -m scripts.report --evaluations reports/main --runs runs --output reports/paper
