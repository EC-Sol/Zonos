FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
RUN pip install uv

RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt update && \
    apt install -y espeak-ng ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./

RUN uv pip install --system nltk pydub
RUN python -m nltk.downloader -d /opt/conda/nltk_data punkt punkt_tab
RUN uv pip install --system -e . && uv pip install --system -e .[compile]
