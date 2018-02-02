# Start with CUDA Torch dependencies 2
FROM kwakinalabs/cuda-torch-cuda-v8
MAINTAINER OBOTUROV Artem

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /model && \
  mkdir /data && \
  mkdir /output && \
  luarocks install tds && \
  luarocks install bit32 

WORKDIR /model/
RUN git clone https://github.com/OpenNMT/OpenNMT --depth 1

# Add the model as the last part to enable re-use
COPY translate.sh /model/translate.sh

WORKDIR /model/OpenNMT

CMD ["/model/translate.sh"]

