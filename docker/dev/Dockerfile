# Dockerfile for development version of Psi4 with adcc

FROM ubuntu
WORKDIR /app
USER root
RUN apt-get update --ignore-missing

# Install python, pip, and other utilities
RUN yes | apt-get install wget
RUN yes | apt-get install git
RUN yes | apt-get install python3
RUN yes | apt-get install python3-pip
RUN yes | apt-get install vim
RUN apt-get -y install htop
RUN apt-get install -y curl

# Install Miniconda
WORKDIR /root
RUN curl "https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh" -o Miniconda3-py37_4.9.2-Linux-x86_64.sh
ENV MINICONDA_INSTALL_DIR /opt/python3.7
RUN bash Miniconda3-py37_4.9.2-Linux-x86_64.sh -b -p $MINICONDA_INSTALL_DIR
ENV PATH ${MINICONDA_INSTALL_DIR}/bin:${PATH}
RUN echo 'export PATH=${MINICONDA_INSTALL_DIR}/bin:${PATH}' >> .bashrc

# Install Psi4
RUN conda create -n psi4-dev -c psi4 psi4
ENV CONDA_DEFAULT_ENV psi4-dev
RUN conda update psi4 -c psi4/label/dev

# Install adcc
RUN apt-get install -y libopenblas-base
RUN apt-get install -y python3-dev build-essential

# SHELL ["conda", "run", "-n", "psi4-dev", "/bin/bash", "-c"]


RUN ["conda", "run", "-n", "psi4-dev", "pip", "install", "pybind11"]
RUN ["conda", "run", "-n", "psi4-dev", "pip", "install", \
        "matplotlib", \
        "scipy", \
        "pandas", \
        "opt_einsum", \
        "adcc", \
        "h5py"]
# RUN pip install pybind11
# RUN pip install matplotlib \
#                 scipy \
#                 pandas \
#                 opt_einsum \
#                 adcc \
#                 h5py

# Install Orquestra runtime dependencies
# RUN pip install pyyaml==5.1

WORKDIR /app

ENTRYPOINT bash
