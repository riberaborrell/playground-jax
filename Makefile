help:
	@echo "---------------HELP-----------------"
	@echo "To create venv with required packages type 'make venv'"
	@echo "To clean the virtual environment type 'make clean'"
	@echo "------------------------------------"

venv:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install --upgrade setuptools
	venv/bin/pip install -e .

clean:
	rm -rf venv
	rm -rf *.egg-info
