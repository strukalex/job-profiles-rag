# Job Profiles Vector Store

A Python application that creates and queries a vector store of job profiles using LangChain, ChromaDB, and HuggingFace embeddings for semantic search capabilities.

## Features

- Processes CSV files containing job profiles
- Creates embeddings using HuggingFace's all-MiniLM-L6-v2 model
- Implements efficient batch processing for large datasets
- Provides similarity search functionality for job profiles

## Prerequisites

- Python 3.8+
- Poetry for dependency management

## Installation

1. Clone the repository
2. Install dependencies using Poetry:

`poetry install`

# Detailed instructions for running in WSL

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

Get poetry environment path:

`poetry env info --path`

Open VS Code Command Palette (Ctrl+Shift+P) and:
- Select "Python: Select Interpreter"
- Choose "Enter interpreter path"
- Paste the Poetry environment path + "/bin/python", e.g. `/home/alstruk/.cache/pypoetry/virtualenvs/job-profiles-rag-gqqwjc62-py3.10/bin/python`

Install kernel for the poetry environment:

`poetry run ipython kernel install --name "job-profiles-rag" --user`

Then in jupyter notebook:
- Click the kernel picker in the top right of the notebook
- Choose the kernel matching your Poetry environment

Add data folder to the root directory, so you have `/data/job profiles/2025-02-07_profiles.csv`

## Usage

The project contains two Jupyter notebooks in the `notebooks` directory:

1. `create_vectore_store.ipynb`: Creates the vector store from job profile data
2. `try_vector_store.ipynb`: Demonstrates how to query the vector store

Run these notebooks to interact with the system.

## Data

The `data/job_profiles` directory contains CSV files with job-related information:
- Classifications
- Job families
- Job profile roles and types
- Organizations
- Profiles
- Scopes
- Streams

## Dependencies

Main dependencies include:
- langchain
- chromadb
- langchain-huggingface
- sentence-transformers
- jupyter

# Kubeernetes Commands
 
## View kubernetes contexts
 
 `kubectl config get-contexts`

## Select context

 `kubectl config use-context docker-desktop`

## Install kubernetes dashboard

`kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.5.1/aio/deploy/recommended.yaml`

`kubectl create serviceaccount admin-user -n kubernetes-dashboard`

`create clusterrolebinding admin-user --clusterrole=cluster-admin --serviceaccount=kubernetes-dashboard:admin-user`

`kubectl -n kubernetes-dashboard create token admin-user`

`kubectl proxy`

Go to: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/

## License

This project is licensed under the terms included in the LICENSE file.

