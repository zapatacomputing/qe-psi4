# Dockerfile for zmachine_psi4:0.0.2

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

# Install development stuff. Not sure if all of these things are necessary.
RUN yes | apt-get install gcc
RUN yes | apt-get install g++
RUN yes | apt-get install python-dev

# Install Psi4
RUN apt-get install -y curl
WORKDIR /root
RUN curl "http://vergil.chemistry.gatech.edu/psicode-download/Psi4conda-1.3.2-py37-Linux-x86_64.sh" -o Psi4conda-1.3.2-py37-Linux-x86_64.sh --keepalive-time 2
RUN bash Psi4conda-1.3.2-py37-Linux-x86_64.sh -b -p $HOME/psi4conda
RUN echo '. $HOME/psi4conda/etc/profile.d/conda.sh' >> ~/.bashrc
RUN echo 'conda activate' >> ~/.bashrc
RUN rm /root/Psi4conda-1.3.2-py37-Linux-x86_64.sh

ENV PATH="/root/psi4conda/bin:${PATH}"
ENV PYTHONPATH="/usr/local/lib/python3.7/dist-packages:${PYTHONPATH}"

RUN python3 -m pip install scipy==1.2.2 \
                           numpy==1.15.4 \
                           openfermion==0.10.0 \
                           pyyaml==5.1 \
                           python-rapidjson==0.9.1

WORKDIR /app

ENTRYPOINT bash
