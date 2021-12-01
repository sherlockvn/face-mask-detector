FROM tensorflow/tensorflow
RUN apt-get update & apt-get upgrade
RUN apt-get -y install zip unzip
RUN apt -y install vim
RUN pip3 install -r requirement.txt
RUN echo "create docker image successfully!"