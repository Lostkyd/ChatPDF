import os
import pickle
import logging
from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader,Settings,PromptTemplate, VectorStoreIndex, Document, ServiceContext, StorageContext, load_index_from_storage

import openai
from llama_index.llms.openai import OpenAI

from langchain_community.llms import HuggingFaceEndpoint
logging.basicConfig(level=logging.DEBUG)


index = None
stored_docs = {}
lock = Lock()

index_name = "./saved_index_dev"
pkl_name = "stored_documents_dev.pkl"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

#Uncomment this for the API key of openAI
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]

# ------- Custom LLM
system_prompt = """ 
+==========================================================================================================+
+ <|SYSTEM|># PDFInstruc                                                                                   +
+ - You are PDFInstruc, a chatbot assistant with immense knowledge.                                        +
+ - You will be given a {query_str} about the content of a PDF File.                                       +
+ - As a chatbot assistant, your task is to provide an answer to the user in succinct and simple language. +
+ - You will use simple, compound, and compound-complex sentences for all your responses.                  +
+==========================================================================================================+
"""

#Uncomment this to use the fine-tuned model and change the model name of original llama 2
# llm = HuggingFaceInferenceAPI(
#     #Fine-tuned llama 2
# model_name="",
#     #Original llama 2
# # model_name = "",
# token="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",
# system_prompt=system_prompt,
# generate_kwargs={"temperature": 0.3, "do_sample": False},
# context_window=4096,
# max_new_tokens=256,
# stopping_ids=[50278, 50279, 50277, 1, 0],
# tokenizer_kwargs={"max_length": 4096}
# )

#Uncomment this for GPT-3.5-turbo model
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.3, system_prompt=system_prompt, api_key="")

embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
)

service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm, embed_model = embed_model)

def initialize_index():
    """Create a new global index, or load one from the pre-set path."""
    global index, stored_docs

def query_index(query_text):
    """Query the global index."""
    global index
    response = index.as_query_engine().query(query_text)
    return response


# def clear_document_content(document):
#     document.text = ""
#     return document

# def clear_file_content(doc_file_path):
#     with open(doc_file_path, "w") as f:
#         f.write("")
#         return
    
def insert_into_index(doc_file_path, doc_id=None):
    """Insert new document into global index."""
    global index, stored_docs
    document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
    if doc_id is not None:
        document.doc_id = doc_id
    with lock:
        if os.path.exists(index_name):
            index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), service_context=service_context)
    # Keep track of stored docs -- llama_index doesn't make this easy
        stored_docs[document.doc_id] = document.text[0:200]  # only take the first 200 chars
        index.insert(document)
        index.storage_context.persist(persist_dir=index_name)
        with open(pkl_name, "wb") as f:
            pickle.dump(stored_docs, f)
    return

def get_documents_list():
    """Get the list of currently stored documents."""
    global stored_docs
    documents_list = []
    for doc_id, doc_text in stored_docs.items():
        documents_list.append({"id": doc_id, "text": doc_text})

    return documents_list


if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_index()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(('localhost', 5602), b'password')
    manager.register('query_index', query_index)
    manager.register('insert_into_index', insert_into_index)
    manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    print("server started...")
    server.serve_forever()