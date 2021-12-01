FROM tensorflow/tensorflow
RUN apt-get update & apt-get upgrade
RUN apt-get -y install zip unzip
RUN apt -y install vim

COPY requirements.txt requirements.txt
RUN pip3 install -r requirement.txt

COPY . .
RUN python3 training_model.py

RUN echo "create docker image successfully!"