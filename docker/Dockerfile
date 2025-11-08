FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    tzdata \
    zip unzip \
    build-essential \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        libgl1 libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
# RUN ln -s /usr/bin/python3.10 /usr/bin/python3

RUN mkdir /app
WORKDIR /app

RUN python -m pip install --upgrade pip && python -m pip install filelock

# ======== torch ========
RUN wget https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp310-cp310-linux_x86_64.whl#sha256=9dde30f399ca22137455cca4d47140dfb7f4176e2d16a9729fc044eebfadb13a && \
    python -m pip install torch-2.5.1+cu124-cp310-cp310-linux_x86_64.whl

RUN wget https://download.pytorch.org/whl/cu124/torchvision-0.20.1%2Bcu124-cp310-cp310-linux_x86_64.whl#sha256=3a055e4e9040b129878d57c39db55f117f975899ff30dd70c8f2621d91170dbe && \
    python -m pip install torchvision-0.20.1+cu124-cp310-cp310-linux_x86_64.whl

RUN wget https://data.pyg.org/whl/torch-2.5.0%2Bcu124/torch_scatter-2.1.2%2Bpt25cu124-cp310-cp310-linux_x86_64.whl && \
    pip install torch_scatter-2.1.2+pt25cu124-cp310-cp310-linux_x86_64.whl

RUN wget https://files.pythonhosted.org/packages/a4/18/1da9464c86c3e59660d40515a93904f9b32726ee62bddb07491a39a743c5/pycuda-2025.1.1.tar.gz \
    && tar xfz pycuda-2025.1.1.tar.gz \
    && cd pycuda-2025.1.1 \
    && python configure.py --cuda-root=/usr/local/cuda-12.4 \
    && su -c "make install"

# ======== requirements ==========

COPY requirements.txt .
RUN pip install -r requirements.txt

# ======== croc ==========

RUN curl https://getcroc.schollz.com | bash

# ======== jupyter ========

RUN pip install jupyterlab==4.2.7 notebook==7.2.2 ipykernel==6.29.5

RUN mkdir /app/notebooks

RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8887" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.notebook_dir = '/'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "jupyter lab --allow-root" > /app/launch_jupyter.sh && \
    chmod +x /app/launch_jupyter.sh

EXPOSE 6011

# =========================

RUN rm requirements.txt