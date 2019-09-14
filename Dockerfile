FROM python:3.7
COPY *.py /home/MLFD/
COPY lib/ /home/MLFD/lib
COPY MLFD_fd_detection/ /home/MLFD/MLFD_fd_detection
COPY figures/ /home/MLFD/figures
ENTRYPOINT ["tail", "-f", "/dev/null"]
