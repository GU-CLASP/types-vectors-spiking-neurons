PYTHON ?= python
PYTTR_SRC ?= $(abspath ../pyttr2/src)
export PYTHONPATH := $(abspath src):$(PYTTR_SRC)$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: check demo web-demo vision-demo gpu-pipeline notebook paths
check:
	$(PYTHON) -m unittest discover -s tests -v

demo:
	$(PYTHON) examples/clevr_ground_truth.py --data-dir data/CLEVR_v1.0

web-demo:
	$(PYTHON) examples/clevr_web.py --data-dir data/CLEVR_v1.0

vision-demo:
	$(PYTHON) examples/clevr_perception.py

gpu-pipeline:
	bash scripts/gpu_clevr_pipeline.sh

notebook:
	$(PYTHON) -m jupyterlab notebooks

paths:
	$(PYTHON) -c 'import pyttr, spinls; print("PyTTR:", pyttr.__file__); print("SPINLS:", spinls.__file__)'
