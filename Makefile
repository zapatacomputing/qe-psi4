include subtrees/z_quantum_actions/Makefile
# Note that that we set PYTHONPATH to ensure that venv site packages take precendence over global site packages

coverage:
	export PYTHONPATH=$$(pwd)/${VENV}/lib/python3.7/site-packages:${PYTHONPATH} && \
		$(PYTHON) -m pytest -m "not integration" \
		--cov=src \
		--cov-fail-under=$(MIN_COVERAGE) tests \
		--no-cov-on-fail \
		--cov-report term-missing \
		&& echo Code coverage Passed the $(MIN_COVERAGE)% mark!

github_actions:
	python3 -m venv ${VENV} && \
		export PYTHONPATH=$$(pwd)/${VENV}/lib/python3.7/site-packages:${PYTHONPATH} && \
		${VENV}/bin/python3 -m pip install --upgrade pip && \
		${VENV}/bin/python3 -m pip install ./z-quantum-core && \
		${VENV}/bin/python3 -m pip install -e '.[develop]'


build-system-deps:
	$(PYTHON) -m pip install setuptools wheel "setuptools_scm>=6.0"
