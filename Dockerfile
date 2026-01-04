FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV HF_HOME=/runpod-volume
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    curl \
    ca-certificates \
    git \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python

# pip bootstrap
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python /tmp/get-pip.py \
 && rm -f /tmp/get-pip.py

RUN python -m pip install --upgrade pip

# 1) Install the matched Torch stack FIRST (cu124)
RUN python -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

# 2) Install your remaining deps (WITHOUT torch/torchvision/torchaudio pins here)
COPY requirements.txt /requirements.txt
RUN python -m pip install --no-cache-dir -r /requirements.txt

# 3) Smoke test (don’t force-import vision subsystems)
RUN python -c "import torch, torchvision; print('torch', torch.__version__, 'vision', torchvision.__version__)"

ADD src .
COPY test_input.json /test_input.json
CMD ["python", "-u", "/handler.py"]
