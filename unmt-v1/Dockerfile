FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
MAINTAINER OBOTUROV Artem

RUN apt-get update && apt-get install -y --no-install-recommends \
  	git \
    binutils \
    ca-certificates \
    cmake-data \
    cpp \
    gcc \
    g++ \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-numpy \
    dh-python \
    dpkg-dev \
    libboost-all-dev \
    cmake \
    wget \
    libopenblas-dev \
    swig \
    && \
	rm -rf /var/lib/apt/lists/* && \
  pip3 install --upgrade pip

RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl

RUN mkdir -p /model

# FastText
WORKDIR /model/
RUN git clone https://github.com/facebookresearch/fastText.git --depth 1 && \
	cd fastText && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j 16 && \
	make install

WORKDIR /model/
# Moses
RUN git clone https://github.com/moses-smt/mosesdecoder.git --depth 1

# MUSE
WORKDIR /model/
RUN git clone https://github.com/aoboturov/MUSE.git --depth 1

# FAISS
WORKDIR /model/
COPY makefile.inc.Linux /model/makefile.inc
ENV BLASLDFLAGS /usr/lib/libopenblas.so.0

RUN git clone https://github.com/facebookresearch/faiss.git --depth 1 && \
  cd faiss && \
  mv /model/makefile.inc /model/faiss/makefile.inc && \
  make -j 16 && \
  make py -j 16 && \
  cp libfaiss.a /usr/lib/ && \
  cd gpu && \
  make -j 16 && \
  make py -j 16 && \
  cp libgpufaiss.a /usr/lib/ && \
  cd .. && \
  cp faiss.py /model && \
  cp swigfaiss.py /model && \
  cp swigfaiss_gpu.py /model && \
  cp _swigfaiss.so /model && \
  cp _swigfaiss_gpu.so /model

ENV PYTHONPATH /model

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /data
RUN mkdir /output

COPY ./ /model/

WORKDIR /model/
RUN pip3 install -r requirements.txt

CMD ["/model/translate.sh", "10", "3", "400", "4"]
