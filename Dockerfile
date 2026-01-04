FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV HF_HOME=/runpod-volume
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    curl \
    ca-certificates \
    git \
    wget \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# make python point to 3.11
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python

# bootstrap pip for python3.11 (avoid ensurepip)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python /tmp/get-pip.py \
 && rm -f /tmp/get-pip.py

# now pip is guaranteed for python3.11
RUN python -m pip install --upgrade pip

COPY requirements.txt /requirements.txt
RUN python -m pip install --no-cache-dir -r /requirements.txt

# torch cu124
RUN python -m pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# pin for infinity-emb's BetterTransformer import path (if you keep it)
RUN python -m pip install --no-cache-dir "optimum<2.0" "transformers<4.49"

RUN python -c "import optimum; import optimum.bettertransformer; import transformers; print('ok', optimum.__version__, transformers.__version__)"

ADD src .
COPY test_input.json /test_input.json
CMD ["python", "-u", "/handler.py"]
