FROM pytorch/pytorch
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
    libboost-all-dev \
    cmake \
    wget \
    libopenblas-dev \
    swig \
    && \
	rm -rf /var/lib/apt/lists/* && \
  pip install --upgrade pip

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

WORKDIR /model/
# OpenNMT-py
RUN git clone https://github.com/OpenNMT/OpenNMT-py --depth 1 && \
  cd OpenNMT-py && \
  pip install -r requirements.txt && \
  python3 setup.py install

WORKDIR /model/
# Moses
RUN git clone https://github.com/moses-smt/mosesdecoder.git --depth 1

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output

COPY clean-corpus-monolingual.perl /model/
COPY translate.sh /model/

CMD ["/model/translate.sh", "0", "3", "2", "200"]
