# FROM python:3.7-alpine
FROM registry.redhat.io/rhel8/python-39:1-27
COPY . /app
WORKDIR /app
# ENV export http_proxy=192.168.2.15:3128
# ENV export https_proxy=192.168.2.15:3128
RUN pip3 install .
CMD ["syd"]
