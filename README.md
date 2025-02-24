# Job Profiles Vector Store

A Python application that creates and queries a vector store of job profiles using LangChain, ChromaDB, and HuggingFace embeddings for semantic search capabilities.

## Features

- Processes CSV files containing job profiles
- Creates embeddings using HuggingFace's all-MiniLM-L6-v2 model
- Implements efficient batch processing for large datasets
- Provides similarity search functionality for job profiles

## Prerequisites

- Python 3.10+
- Poetry for dependency management

## Installation

1. Clone the repository
2. Install dependencies using Poetry:

`poetry install`

3. Register for a LLM API, such as Mistral Small through Azure Foundry
4. Rename .env.sample into sample and set your environment variables
5. Copy data into the root folder so you have `/data/job profiles/2025-02-07_profiles.csv`


# Running Open WebUI with local Kubernetes

Follow these instructions to run a UI that allows you to run queries with the RAG pipeline. Before getting started, ensure you have `job_profiles_db2` in the root by building the vector store through the `create_vectore_store.ipynb` notebook (see below).

## Build and run API docker image
If dependencies changed, generate requirements.txt from poetry and copy to /backend:
`poetry export --format=requirements.txt --output requirements.txt --without-hashes`

Build docker image (run from root):

`docker build -f backend/Dockerfile -t rag-backend:local .`

To run the api docker image:

`docker run --env-file ./.env -p 8000:8000 rag-backend:local`


## Setup Kubernetes

Select local context:
`kubectl config use-context docker-desktop`

To view contexts:
`kubectl config get-contexts`

Add nginx ingress controller:
`kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.0/deploy/static/provider/cloud/deploy.yaml`

Check that ingress controller is working:
`kubectl get pods -n ingress-nginx`

Create namespace:
`kubectl create namespace fd34fb-dev`

Apply local overlay configuration:
`kubectl apply -k kubernetes/openwebui/overlays/local`

Open WebUI should be available at (ignore security warning):
`https://kubernetes.docker.internal/`

Go to `Admin Panel -> Settings -> Connections`, delete all records. Under `OpenAI API` add a new record with `http://external-api:8000/v1` for URL and
anything for the `Key`

The model should now be available on the main selection screen and querying should work.

## Install kubernetes dashboard

`kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.5.1/aio/deploy/recommended.yaml`

`kubectl create serviceaccount admin-user -n kubernetes-dashboard`

`kubectl create clusterrolebinding admin-user --clusterrole=cluster-admin --serviceaccount=kubernetes-dashboard:admin-user`

`kubectl -n kubernetes-dashboard create token admin-user`

`kubectl proxy`

Go to: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/

## Notebooks Usage

The project contains two Jupyter notebooks in the `notebooks` directory:

1. `create_vectore_store.ipynb`: Creates the vector store from job profile data
2. `try_vector_store.ipynb`: Demonstrates how to query the vector store


# To run notebooks in WSL

Update python:

`sudo apt update && sudo apt upgrade`

`sudo apt upgrade python3`

Install poetry:

`curl -sSL https://install.python-poetry.org | python3 -`

`echo 'export PATH="/home/<USER_NAME>/.local/bin:$PATH"' >> ~/.bashrc`

`source ~/.bashrc`

Add vs code to path in .bashrc:

`export PATH="/mnt/c/Users/<USER_NAME>/AppData/Local/Programs/Microsoft VS Code/bin:$PATH"`

If you get `/usr/bin/env: ‘sh’: No such file or directory`, ensure you have this in .bashrc:

`export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH`

Then navigate to ex.:

`cd /mnt/c/Users/<USER_NAME>/GitHub/job-profiles-rag`

Start vs code:

`code .`

Install WSL extension for vs code and restart vs code.

Install Python extension (should show up under WSL: UBUNTU - INSTALLED)

Run `poetry install`

If you get `[Errno 2] No such file or directory: 'python`, you may need to create a symlink from python3 to python:

`sudo ln -s /usr/bin/python3 /usr/bin/python`

Get poetry environment path:

`poetry env info --path`

Open VS Code Command Palette (Ctrl+Shift+P) and:
- Select "Python: Select Interpreter"
- Choose "Enter interpreter path"
- Paste the Poetry environment path + "/bin/python", e.g. `/home/<USER_NAME>/.cache/pypoetry/virtualenvs/job-profiles-rag-gqqwjc62-py3.10/bin/python`

Install kernel for the poetry environment:

`poetry run ipython kernel install --name "job-profiles-rag" --user`

Then in jupyter notebook:
- Click the kernel picker in the top right of the notebook
- Choose the kernel matching your Poetry environment

Add data folder to the root directory, so you have `/data/job profiles/2025-02-07_profiles.csv`

# Docker - other

## Open WebUI in docker
To run open webui in docker such that it can connect to api running on localhost:

`docker run -d -p 3000:8080   --add-host=host.docker.internal:host-gateway   -e OPENAI_API_BASE_URL=http://host.docker.internal:8000   -v open-webui:/app/backend/data   --name open-webui   --restart always   ghcr.io/open-webui/open-webui:main`

In open webui set api url to:

`http://host.docker.internal:8000/v1`

## To update local openwebui image:

`docker pull ghcr.io/open-webui/open-webui:main`

Stop and remove the current container
`docker rm -f open-webui`

## License

This project is licensed under the terms included in the LICENSE file.


### Show current packages vs latest versions
poetry show --latest

### Show debug code for langchan processes
```
from langchain.globals import set_debug
set_debug(True)
```

### Debug everything
import logging
logging.basicConfig(level=logging.DEBUG)

### Add a shortcut to execute in the current Debug Console:

Open keyboard shortcuts: `Ctrl+K+S`. Search for `Debug: Evaluate in Debug Console` and select a shortcut, like `Alt+Shift+E`

### To set custom graphics

```
docker cp backend/assets/splash.png open-webui:/app/backend/open_webui/static/splash.png

docker cp backend/assets/square/android-chrome-192x192.png open-webui:/app/backend/open_webui/static/favicon.png

docker cp backend/assets/square/android-chrome-512x512.png open-webui:/app/backend/open_webui/static/logo.png

docker cp backend/assets/square/favicon.ico open-webui:/app/build/favicon/favicon.ico

docker cp backend/assets/square/android-chrome-192x192.png open-webui:/app/build/favicon/favicon-96x96.png
```

### To run openwebui with built-in ollama service:

`docker run -d -p 3000:8080 --gpus all -v ollama:/root/.ollama -v open-webui:/app/backend/data --name open-webui-gpu --restart always ghcr.io/open-webui/open-webui:ollama`

Data will be contained in `/root/.ollama/app/backend/data`

Restart ollama:

`taskkill /F /IM "ollama app.exe" /IM ollama.exe`
`ollama serve`

`ollama list`: See models
`ollama ps`: Displays currently running models
`ollama stop`: Stops a running model
`ollama rm`: Removes a model from the system

To run a new model from HF:

`ollama run hf.co/username/repository`

Enabled mirrored mode in WSL (allows treating localhost to connect to services running on windows):

Add `.wslconfig` in `C:\Users\user_name` with:

```
[wsl2]
networkingMode=mirrored
```

### To run neo4j in docker:

```
docker run \
    --name neo4j-db \
    --publish=7474:7474 --publish=7687:7687 \
    --volume=$HOME/neo4j/data:/data \
    --volume=$HOME/neo4j/logs:/logs \
    --env NEO4J_AUTH=neo4j/your_password \
    --restart always \
    neo4j:latest
```

Then login here: `http://localhost:7474/`

### Enable wrapping in jupyter notebooks in vscode:

Open VS Code Settings (⌘ + , on Mac / Ctrl + , on Windows).

Search for "Notebook > Output: Word Wrap".

Check the box to enable this setting or add to settings.json:

```
"notebook.output.wordWrap": true
```

