FROM python:3.5-slim
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	&& rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

RUN pip install numpy langdetect

RUN mkdir -p /model/
WORKDIR /model/



RUN mkdir /data
RUN mkdir /output
COPY mosesdecoder /model/mosesdecoder
COPY vocabs /model/vocabs 
COPY translate.sh /model/
COPY tokenizer.sh /model/
COPY detokenizer.sh /model/
COPY word-by-word.py /model/
COPY lang_detect.py /model/

CMD ["/model/translate.sh"]
