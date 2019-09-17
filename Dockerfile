FROM python:3.7
COPY *.py /home/MLFD/
COPY lib/ /home/MLFD/lib
COPY MLFD_fd_detection/ /home/MLFD/MLFD_fd_detection
COPY figures/ /home/MLFD/figures
RUN pip install datawig anytree pandas
RUN apt update
RUN apt install -y screen htop
ENTRYPOINT ["tail", "-f", "/dev/null"]
