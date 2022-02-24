# Dockerfile for development version of Psi4 with adcc

FROM ubuntu
WORKDIR /app
USER root
RUN apt-get update --ignore-missing


# Install Python 3.7
RUN apt-get install -y software-properties-common && \
        add-apt-repository -y ppa:deadsnakes/ppa && \
        apt-get install -y python3.7 && \
        apt-get install -y python3-pip && \
        apt-get install -y python3.7-dev

# Install other utilities
RUN apt-get -y install \
        wget \
        git \
        vim \
        htop \
        curl 

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

RUN ["conda", "run", "-n", "psi4-dev", "pip", "install", "pybind11"]
RUN ["conda", "run", "-n", "psi4-dev", "pip", "install", \
        "matplotlib", \
        "scipy", \
        "pandas", \
        "opt_einsum", \
        "adcc", \
        "h5py"]

RUN conda init bash
RUN echo "conda activate psi4-dev" >> /root/.bashrc

WORKDIR /app

ENTRYPOINT bash
