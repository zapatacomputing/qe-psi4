import warnings
import site
import sys
import setuptools

try:
    from subtrees.z_quantum_actions.setup_extras import extras
except ImportError:
    warnings.warn("Unable to import extras")
    extras = {}

# Workaound for https://github.com/pypa/pip/issues/7953
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="qe-psi4",
    use_scm_version=True,
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Psi4 integration for Quantum Engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/qe-psi4",
    packages=setuptools.find_packages(where="src/python"),
    package_dir={"": "src/python"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["openfermion>=1.0.0", "numpy>=1.20"],
    extras_require=extras,

)
