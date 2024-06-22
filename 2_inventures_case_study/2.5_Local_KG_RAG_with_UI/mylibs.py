# Importing libraries
import os
import pickle
import json # saving sessions data in a json file
# from google.colab import userdata # for importing env variable on Colab
import matplotlib.pyplot as plt
import tiktoken # for counting the # of tokens in each document
import numpy as np
import pandas as pd
import umap # for dimensionality reduction
from sklearn.mixture import GaussianMixture # for clustering
from typing import Dict, List, Optional, Tuple # for type hinting

from bs4 import BeautifulSoup # for scraping and parsing

# Wikiloader in case we want to retrieve data from Wikipedia as a trainset
from langchain_community.document_loaders import WikipediaLoader
# converting doc_text files into langchain document objects
from langchain.docstore.document import Document

from langchain_openai import OpenAIEmbeddings # For text-to-vector
from langchain_openai import ChatOpenAI # For API call to LLM
from langchain_anthropic import ChatAnthropic # Alternative to OpenAI LLM

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

RANDOM_SEED = 307  # Fixed seed for reproducibility of clustering

from langchain_community.vectorstores import Chroma #In-memory VectorDB

from langchain import hub # Langchain Hub for RAG prompt templates
# placeholder used in chain
from langchain_core.runnables import RunnablePassthrough
import streamlit as st



#### Knowledge Graph imports
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI

# Make sure to `pip install openai` first
embedding_server = OpenAI(base_url="http://localhost:1234/v1/", api_key="lm-studio")

class TrueOpenAIEmbeddings:
    def __init__(self, model="nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"):
        self.model_type = model

    def embed_query(self, text):
        # print(f"inside embed sigle and input is {text}")
        text = text.replace("\n", " ")
        response = embedding_server.embeddings.create(input=[text], model=self.model_type)
        return response.data[0].embedding

    def embed_documents(self, texts):
        # print(f"inside embed documents and input is {texts}")
        return [self.embed_query(d) for d in texts]
