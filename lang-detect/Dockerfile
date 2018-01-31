FROM python:3.5-slim
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

RUN pip install langdetect

RUN mkdir -p /model/
WORKDIR /model/

RUN mkdir /data
RUN mkdir /output

COPY translate.sh /model/
COPY lang_detect.py /model/

CMD ["/model/translate.sh"]
