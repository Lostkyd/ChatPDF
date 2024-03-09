import os
import pickle
# NOTE: for local testing only, do NOT deploy with your key hardcoded
# os.environ['OPENAI_API_KEY'] = "your key here"

from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index.core import Settings,SimpleDirectoryReader,PromptTemplate, VectorStoreIndex, Document, ServiceContext, StorageContext, load_index_from_storage

# ---- added
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
import logging
# --- added

index = None
stored_docs = {}
lock = Lock()

index_name = "./saved_index"
pkl_name = "stored_documents.pkl"

def initialize_index():
    """Create a new global index, or load one from the pre-set path."""
    global index, stored_docs
    logging.basicConfig(level=logging.DEBUG) 
    # ------- Custom LLM

    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """

    # This will wrap the default prompts that are internal to llama-index
    query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.2, "do_sample": True},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="Lostkyd/llama-2-7b-docunstruc",
        model_name="Lostkyd/llama-2-7b-docunstruc",
        device_map="auto",
        stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 1042},
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs = {"torch_dtype": torch.float32}
    )
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    )
    # # ------- Custom LLM
    Settings.llm = llm
    Settings.chunk_size = 1042
    #service_context = ServiceContext.from_defaults(chunk_size_limit=1042, llm=llm)
    with lock:
        if os.path.exists(index_name):
            index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), llm = llm, chunk_size = 1042, embed_model=embed_model)
        else:
            index = VectorStoreIndex([], llm = llm, chunk_size = 1042, embed_model=embed_model)
            index.storage_context.persist(persist_dir=index_name)
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                stored_docs = pickle.load(f)


def query_index(query_text):
    """Query the global index."""
    global index
    response = index.as_query_engine().query(query_text)
    return response


def insert_into_index(doc_file_path, doc_id=None):
    """Insert new document into global index."""
    global index, stored_docs
    document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
    if doc_id is not None:
        document.doc_id = doc_id

    with lock:
        # Keep track of stored docs -- llama_index doesn't make this easy
        stored_docs[document.doc_id] = document.text[0:200]  # only take the first 200 chars

        index.insert(document)
        index.storage_context.persist(persist_dir=index_name)
        
        with open(pkl_name, "wb") as f:
            pickle.dump(stored_docs, f)

    return

def get_documents_list():
    """Get the list of currently stored documents."""
    global stored_doc
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