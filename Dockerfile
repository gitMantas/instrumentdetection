#FROM python:3.7.6-stretch
from nvidia/cuda
#FROM 

#MAINTAINER 

# install build utilities
RUN apt-get update && \
	apt-get install -y python3 python3-pip gcc make apt-transport-https ca-certificates build-essential

# check our python environment
RUN python3 --version
RUN pip3 --version

RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

# Installing python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY ./* /home/appuser
#RUN ls -la /src/*

# Running Python Application
#CMD ["python3", "/src/main.py"]
