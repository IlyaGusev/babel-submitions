# Start with CUDA Torch dependencies 2
FROM kaixhin/torch:latest
MAINTAINER OBOTUROV Artem

# playground will contain user defined scripts, it should be run as:
# docker run -v `pwd`:/data -it basel-baseline
RUN mkdir /model && \
  mkdir /data && \
  mkdir /output && \
  luarocks install tds && \
  luarocks install bit32

WORKDIR /model/
RUN git clone https://github.com/OpenNMT/OpenNMT

# Add the model as the last part to enable re-use
COPY translate.sh /model/translate.sh

CMD ["/model/translate.sh"]
