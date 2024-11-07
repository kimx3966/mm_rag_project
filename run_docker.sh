export LOCAL_NIM_CACHE=~/.cache/nim
export TRANSFORMERS_CACHE=~/.cache/huggingface/models
mkdir -p "$LOCAL_NIM_CACHE"
mkdir -p "$TRANSFORMERS_CACHE"
sudo docker run -it --rm \
    --gpus 0 \
    --shm-size=16GB \
    -e NGC_API_KEY=$NGC_API_KEY \
    -e TRANSFORMERS_CACHE=/opt/nim/.cache/huggingface/models \
    -v "$TRANSFORMERS_CACHE:/opt/nim/.cache/huggingface/models" \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:latest