FROM python:latest

# OS
RUN apt-get update

# Python package
RUN python -m pip install -U pip
RUN pip install numpy jupyter matplotlib tensorflow pillow

# Aliases
RUN echo "alias py=\"python\"" >> ~/.bashrc
RUN echo "alias python=\"python3.12\"" >> ~/.bashrc