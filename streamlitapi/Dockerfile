FROM  nvidia/cuda:11.0-runtime-ubuntu20.04
WORKDIR /home/user/streamlit/

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3.7-distutils python3-pip curl wget git pkg-config cmake swig

# Set default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --set python3 /usr/bin/python3.7

COPY ./requirements.txt /home/user/streamlit
COPY ./peltarion_logotype_vertical.svg /home/user/streamlit
RUN pip install -r requirements.txt
COPY ./webapp.py /home/user/streamlit
COPY ./score_objects.pkl /home/user/streamlit
EXPOSE 8501
CMD ["streamlit", "run", "webapp.py"]
