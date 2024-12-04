#########################################################################################
FROM alpine:3.20

VOLUME /data
WORKDIR /data

ARG BUILD_VERSION=0.1
ARG DATA_PACK_DIR=
ENV DATA_PACK_DIR=${DATA_PACK_DIR}

RUN test -n "${DATA_PACK_DIR}" || (echo "ERROR: DATA_PACK_DIR is not set or empty" && exit 1)

COPY ${DATA_PACK_DIR} /data/

RUN ls -lah /data

ENTRYPOINT [""]

CMD ["ls", "-lah", "/data"]    
