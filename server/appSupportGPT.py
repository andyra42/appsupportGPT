# import imports an entire code library. from import imports a specific member or members of the library.
import logging
import os

# Shutil module offers high-level operation on a file like a copy, create, and remote operation on the file
import shutil

# Python subprocess module is a tool that allows you to run other programs or commands from your Python code
import subprocess

# PyTorch is an open source machine learning (ML) framework based on the Python programming language and the Torch
# library. Torch is an open source ML library used for creating deep neural networks and is written in the Lua
# scripting language.
import torch

# AutoGPTQ library in the Transformers ecosystem allows users to quantize and run LLMs using the GPTQ algorithm.
from auto_gptq import AutoGPTQForCausalLM

# Flask is a widely used micro web framework for creating APIs in Python.
from flask import Flask, jsonify, request

# Chains is an incredibly generic concept which returns to a sequence of modular components (or other chains)
# combined in a particular way to accomplish a common use case. The RetrievalQAChain is a chain that combines a
# Retriever and a QA chain . It is used to retrieve documents from a Retriever and then use a QA
# chain to answer a question based on the retrieved documents.
from langchain.chains import RetrievalQA

# The Hugging Face Inference API allows us to embed a dataset using a quick POST call easily. Since the embeddings
# capture the semantic meaning of the questions, it is possible to compare different embeddings and see how different
# or similar they are
from langchain.embeddings import HuggingFaceInstructEmbeddings

# hf_hub_download returns the local path where the model was downloaded
from huggingface_hub import hf_hub_download

# The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most
# of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
# Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering
from langchain.llms import HuggingFacePipeline, LlamaCpp

# Prompt templates are pre-defined recipes for generating prompts for language models. A template may include
# instructions, few-shot examples, and specific context and questions appropriate for a given task. LangChain
# provides tooling to create and work with prompt templates.
from langchain.prompts import PromptTemplate

# A vector store takes care of storing embedded data and performing vector search
# Chroma is a vector store and embeddings database designed from the ground-up to make it easy to build AI
# applications with embeddings
from langchain.vectorstores import Chroma

# Transformers provides APIs to quickly download and use those pretrained models on a given text,
# fine-tune them on your own datasets and then share them with the community
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
SHOW_SOURCES = True
logging.info(f"Running on: {DEVICE_TYPE}")
logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            # The code supports all huggingface models that ends with GPTQ and have some variation
            # of .no-act.order or .safetensors in their HF repo.
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
            device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    print("Main Application")
