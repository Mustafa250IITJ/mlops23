# From  ubuntu:22.04
FROM python:3.11.4

COPY . /digits/
# 
# RUN apt-get update
# RUN apt-get install -y python3
RUN pip3 install -r ./digits/requirements.txt

WORKDIR /digits

VOLUME /digits/model

# CMD python exp.py 3
CMD ["python","train_model.py"]
# CMD ["pytest"]
# CMD ["echo", "1st cmd line"