# Dockerfile for zmachine_psi4:0.0.2

FROM ubuntu
WORKDIR /app
USER root

ARG PSI4_VERSION="Psi4conda-1.3.2-py37-Linux-x86_64"

# Install python, pip, and other utilities
RUN apt-get update -y --ignore-missing && \
    apt-get install -y curl git

# Install Psi4
RUN curl "http://vergil.chemistry.gatech.edu/psicode-download/${PSI4_VERSION}.sh" -o /root/${PSI4_VERSION}.sh --keepalive-time 2 && \
    bash /root/${PSI4_VERSION}.sh -b -p $HOME/psi4conda && \
    echo ". $HOME/psi4conda/etc/profile.d/conda.sh\nconda activate" >> ~/.bashrc && \ 
    rm /root/${PSI4_VERSION}.sh

ENV PATH="/root/psi4conda/bin:${PATH}"
ENV PYTHONPATH="/root/psi4conda/lib/python3.7/site-packages:/usr/local/lib/python3.7/dist-packages:${PYTHONPATH}"

RUN python3 -m pip install --upgrade pip==20.2.4 && \ 
    python3 -m pip install scipy==1.2.2 \
    numpy==1.15.4 \
    openfermion==0.10.0 \
    pyyaml==5.1 \
    python-rapidjson==0.9.1

WORKDIR /app

ENTRYPOINT bash
