FROM tensorflow/tensorflow
RUN apt-get update & apt-get upgrade
RUN apt-get install zip unzip -y --fix-missing
RUN apt-get install vim -y --fix-missing

COPY requirements.txt requirements.txt
RUN pip3 install -r requirement.txt

COPY . .
RUN python3 training_model.py

RUN echo "create docker image successfully!"