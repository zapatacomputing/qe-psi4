include subtrees/z_quantum_actions/Makefile
# Note that that we set PYTHONPATH to ensure that venv site packages take precendence over global site packages

github_actions:
	python3 -m venv ${VENV} && \
		${VENV}/bin/python3 -m pip install --upgrade pip && \
		${VENV}/bin/python3 -m pip install ./z-quantum-core && \
		${VENV}/bin/python3 -m pip install -e '.[develop]'