include subtrees/z_quantum_actions/Makefile

github_actions:
	apt-get update
	apt-get install -y python3.7-venv

	python3 -m venv ${VENV} && \
		${VENV}/bin/python3 -m pip install --upgrade pip && \
		${VENV}/bin/python3 -m pip install ./z-quantum-core && \
		${VENV}/bin/python3 -m pip install -e '.[develop]'