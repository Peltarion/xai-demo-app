FROM  nvidia/cuda:11.0-runtime-ubuntu20.04
WORKDIR /home/user/deploy/

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip curl wget git pkg-config cmake swig

# Set default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --set python3 /usr/bin/python3.7
COPY ./requirements.txt /home/user/deploy/
COPY ./app.py /home/user/deploy/
COPY ./score_objects.pkl /home/user/deploy/
COPY ./helper.py /home/user/deploy/
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python3", "app.py"]
