
# LLM Assistant and Attack Platform

## Quick Summary

In this repository, we present a unique website hosting an LLM assistant designed to facilitate understanding and experimenting with LLMs. Users can engage with our LLM chatbot, attempting various Prompt Engineering attacks to discern vulnerabilities or strengths within different LLM frameworks. Our system provides feedback on the success of these attempted attacks, offering an interactive learning and research environment.

## Background

Our research focuses on the integration of various LLMs (including GPT-3.5, GPT-4.0, and different fine-tuned versions of Mistral 7b) with our Innovative Intelligence Concentration Framework (IC Framework) for Retrieval Augmented Generation (RAG). This framework is designed to enhance the LLMs' contextually relevant responses, thereby improving user interaction and information accuracy.

We have 2 submitted papers about this matter and will be linking them here once they are published

We've applied the concept of 'attack LLM' as a central theme, incorporating numerous sources on how to conduct such attacks into our Milvus vector database for effective RAG with the LLM. 


<img width="619" alt="Screenshot 2024-03-04 at 12 13 25 PM" src="https://github.com/GrandRecs/ic_agent/assets/66585292/4cd7fcc0-5e80-4625-8fd1-8e84827e26b6">


## Implementation Details

Our setup includes a Neural Chat 7b model and GPT-4.0, both integrated with and without our IC Framework, serving as the LLM assistant. For attack simulations, we employ the Zypher 7b model, inviting students to explore various attack methodologies. Authentication is managed via Shibboleth, provided by the University of Toronto.

## Requirements

To use our platform, ensure your system meets the following specifications:

- Python >= 3.10
- Node.js == v20.11.1
- GIT LFS
#### For Windows and Linux
- CUDA > 12.1
- nvidia-docker2
- A compatible GPU, VRAM>=8GB
#### For MacOS
- M-series Apple silicon, RAM>=16GB


## How to use

```
git clone https://github.com/GrandRecs/ic_agent.git
mkdir models
cd models
git lfs install
git clone https://huggingface.co/Intel/neural-chat-7b-v3-3
git clone https://huggingface.co/BAAI/bge-base-en-v1.5
```
download the llm and embeding model from huggingface and place them into models folder,
additionally you may need to set proper permissions for volumes/mongodb to allow mongondb to write to the folder


### For windows and ubuntu
```
cp docker-compose-nvidia.yml docker-compose.yml
docker compose up -d
```

### For MACOS (BETA)
```
cd models && mkdir gguf && cd gguf
wget https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF/resolve/main/neural-chat-7b-v3-3.Q4_K_M.gguf
mv neural-chat-7b-v3-3.Q4_K_M.gguf neural-chat-7b-v3-3.gguf
```
download additional gguf model from huggingface and place them into gguf models/gguf folder

start the llm server(recommend using a conda virtual environment)
```
cd mps_runtime
conda create -n llm python=3.10
conda activate llm
pip3 uninstall ctransformers --yes
CT_METAL=1 pip3 install ctransformers --no-binary ctransformers
pip3 install -r requirements.txt
screen python3 llmserver.py # you may wish to use screen to run the server in the background
```

start the program
```
cp docker-compose-mps.yml docker-compose.yml
docker compose up -d
```

visit localhost:8080 to access the service

## Conclusion

Our platform demonstrates the practical application of hosting an LLM for educational purposes, allowing educators and students to interact with advanced AI models. The integration of RAG aims to refine the LLM's responses, reducing misinformation and enhancing the educational value.

