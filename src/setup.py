import setuptools
import os

readme_path = os.path.join("..", "README.md")
with open(readme_path, "r") as f:
    long_description = f.read()

setuptools.setup(
    name="qe-psi4",
    version="0.1.0",
    author="Zapata Computing, Inc.",
    author_email="info@zapatacomputing.com",
    description="Psi4 integration for Quantum Engine.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zapatacomputing/qe-psi4",
    packages=["qepsi4"],
    package_dir={"": "python"},
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "z-quantum-core",
    ],
)
