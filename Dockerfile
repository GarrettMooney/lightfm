FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y build-essential libxml2 libxslt-dev wget bzip2 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    else \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    fi && \
    wget --quiet $MINICONDA_URL -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda install -y --override-channels -c conda-forge pytest jupyter scikit-learn && \
    conda clean -afy

ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/lightfm

COPY . .

RUN pip install . && \
    cp -r tests /home/tests && \
    cp -r examples /home/examples && \
    rm -rf /home/lightfm

WORKDIR /home
