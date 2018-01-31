FROM python:3.6-slim
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get install -y --no-install-recommends \
  	git \
    binutils \
    ca-certificates \
    cmake-data \
    cpp \
    gcc \
    g++ \
    dh-python \
    dpkg-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-numpy \
    libboost-all-dev \
    cmake \
    wget \
    libopenblas-dev \
    swig \
    && \
	rm -rf /var/lib/apt/lists/* && \
  pip3 install --upgrade pip

RUN mkdir -p /model

# FastText
WORKDIR /model/
RUN git clone https://github.com/facebookresearch/fastText.git --depth 1 && \
	cd fastText && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make && \
	make install

# MUSE
WORKDIR /model/
RUN git clone https://github.com/facebookresearch/MUSE.git --depth 1

# FAISS
WORKDIR /model/
COPY makefile.inc.Linux /model/makefile.inc
ENV BLASLDFLAGS /usr/lib/libopenblas.so.0

RUN git clone https://github.com/facebookresearch/faiss.git --depth 1 && \
  cd faiss && \
  mv /model/makefile.inc /model/faiss/makefile.inc && \
  make && \
  make py

# OpenNMT-py
RUN git clone https://github.com/OpenNMT/OpenNMT-py --depth 1 && \
  cd OpenNMT-py && \
  pip3 install -r requirements.txt && \
  python3 setup.py install

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output

CMD ["/model/translate.sh"]
