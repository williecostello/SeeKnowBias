# Create common environment
FROM ubuntu:18.04 as conda-env

# Set character encoding environment variables
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# Allow apt-get install without interaction from console
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory to /root
WORKDIR /root

# System updates and configurations
RUN apt-get update && apt-get -y --no-install-recommends install \
		ca-certificates \
		git \
		ssh \
		wget && \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
	bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda && \
	rm Miniconda3-latest-Linux-x86_64.sh

# Set the path env to include miniconda
ENV PATH /root/miniconda/bin:$PATH

# Serving environment
FROM conda-env 

# Copy over dependency file to cache (hopefully) core packages
COPY ./requirements.txt /root/requirements.txt


# Pip install 
RUN pip install --upgrade pip setuptools && \
	pip install -r /root/requirements.txt --no-cache-dir

COPY . /root


# Download trained BERT pytorch model
RUN wget "https://docs.google.com/uc?export=download&id=1x__YR6PO6_cW-vy3vb0cdjWRYqjBqPW0" -O /root/model.pkl

EXPOSE 80 8051

ENTRYPOINT [ "streamlit", "run"]
CMD ["/root/app.py"]