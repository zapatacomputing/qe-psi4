include subtrees/z_quantum_actions/Makefile

coverage:
	$(PYTHON) -m pytest -m "not integration" \
		--cov=src \
		--cov-fail-under=$(MIN_COVERAGE) tests \
		--no-cov-on-fail \
		--cov-report term-missing \
		&& echo Code coverage Passed the $(MIN_COVERAGE)% mark!

github_actions:
	# Make sure that venv site packages take precendence over global site packages
	export PYTHONPATH=${VENV}/lib/python3.7/site-packages:${PYTHONPATH}

	python3 -m venv ${VENV} && \
		${VENV}/bin/python3 -m pip install --upgrade pip && \
		${VENV}/bin/python3 -m pip install ./z-quantum-core && \
		${VENV}/bin/python3 -m pip install -e '.[develop]'