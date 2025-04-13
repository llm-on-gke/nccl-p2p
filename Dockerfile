
FROM  nvcr.io/nvidia/pytorch:24.12-py3
WORKDIR /p2p-test
RUN apt-get -y update
RUN apt-get -y install iputils-ping numactl pciutils
COPY p2p-test.py ./

ENTRYPOINT ["/bin/bash"]
