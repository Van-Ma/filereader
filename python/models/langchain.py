# models/langchain.py
# This file contains the stateful, KV-caching chat implementation.

import logging
from typing import Any, Optional, List, Dict, Sequence, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.chat_models import ChatHuggingFace

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

from models.custom_huggingface_llm import CustomHuggingFaceLLM
from models.model_parameters import ModelParameters, HuggingFaceParameters
from models.model_cache import GLOBAL_HF_MODEL_CACHE

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define LangGraph State
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]
    file_contents: Annotated[List[str], "append_file_content"]
    rag_enabled: bool

# Helper function to append file content to state (for LangGraph Annotated)
def append_file_content(current: List[str], new: List[str]) -> List[str]:
    if current is None:
        current = []
    return current + new

# Allowed model names for HuggingFace backend
ALLOWED_HUGGINGFACE_MODELS = {
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# --- Global LangGraph App Instances ---
_global_base_app = None
_global_rag_app = None

class LangChainBase:
    """A base LangChain implementation using LangGraph for state management and a custom LLM."""
    
    def __init__(self, model_params: ModelParameters, **kwargs):
        # Decentralized Validation for LangChainBase
        framework_type = model_params["framework_type"]
        backend = model_params["backend"]
        model_version = model_params["model_version"]
        hf_params = model_params.get("hf_params")

        if framework_type != "LangChain":
            raise ValueError(f"Invalid framework_type for LangChainBase: '{framework_type}'. Must be 'LangChain'.")
        if backend != "HuggingFace":
            raise ValueError(f"Invalid backend for LangChainBase: '{backend}'. Must be 'HuggingFace' for now.")
        if model_version != "Base":
            raise ValueError(f"Invalid model_version for LangChainBase: '{model_version}'. Must be 'Base'.")
        
        if not hf_params or not isinstance(hf_params, dict):
            raise ValueError("Missing or invalid 'hf_params' for HuggingFace backend in LangChainBase.")
        hf_params_typed: HuggingFaceParameters = hf_params 

        if hf_params_typed["model_name"] not in ALLOWED_HUGGINGFACE_MODELS:
            raise ValueError("Invalid model_name for LangChainBase: '{}'. Must be one of {}.".format(hf_params_typed["model_name"], list(ALLOWED_HUGGINGFACE_MODELS)))

        self.model_id = hf_params_typed["model_name"]
        self.use_kv_cache = hf_params_typed["use_kv_cache"]

        if self.use_kv_cache:
            self.llm = CustomHuggingFaceLLM(hf_params=hf_params_typed)
        else:
            logging.info("Initializing ChatHuggingFace with HuggingFacePipeline model '{}' (no explicit KV cache across invocations).".format(self.model_id))
            try:
                if self.model_id in GLOBAL_HF_MODEL_CACHE:
                    logging.info(f"Using cached HuggingFacePipeline model for '{self.model_id}'.")
                    loaded_model = GLOBAL_HF_MODEL_CACHE[self.model_id]['model']
                    loaded_tokenizer = GLOBAL_HF_MODEL_CACHE[self.model_id]['tokenizer']
                else:
                    logging.info(f"Loading HuggingFacePipeline model '{self.model_id}' (not in cache)...")
                    if torch.cuda.is_available():
                        device = torch.device("cuda")
                        logging.info("CUDA available. Loading model on GPU.")
                        to_kwargs = {"device": device}
                    else:
                        logging.info("CUDA not available. Loading model on CPU.")
                        to_kwargs = {}  # .to() call will be skipped for CPU to avoid negative index

                    loaded_tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                    loaded_model = AutoModelForCausalLM.from_pretrained(
                        self.model_id, torch_dtype=torch.bfloat16
                    )
                    if to_kwargs:
                        loaded_model = loaded_model.to(**to_kwargs)
                    if loaded_tokenizer.pad_token is None:
                        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token
                    
                    GLOBAL_HF_MODEL_CACHE[self.model_id] = {
                        'model': loaded_model,
                        'tokenizer': loaded_tokenizer
                    }
                    logging.info(f"HuggingFacePipeline model and tokenizer for '{self.model_id}' cached.")

                from transformers import pipeline as hf_pipeline

                gen_kwargs = dict(max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9)
                if torch.cuda.is_available():
                    pipe = hf_pipeline(
                        "text-generation",
                        model=loaded_model,
                        tokenizer=loaded_tokenizer,
                        device=0,
                        **gen_kwargs,
                    )
                else:
                    pipe = hf_pipeline(
                        "text-generation",
                        model=loaded_model,
                        tokenizer=loaded_tokenizer,
                        **gen_kwargs,
                    )

                pipeline_llm = HuggingFacePipeline(pipeline=pipe)
                self.llm = ChatHuggingFace(llm=pipeline_llm)
            except Exception as e:
                logging.error("Failed to load HuggingFacePipeline model '{}'. This could be due to: ".format(self.model_id) +
                              "1. Invalid model name."
                              "2. Insufficient memory/GPU resources."
                              "3. Network issues preventing model download."
                              "4. Hugging Face rate limits or authentication issues (for private models)."
                              "Error details: {}".format(e))
                raise

        self.base_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly chatbot."),
            MessagesPlaceholder(variable_name="messages")
        ])

    def _call_model_node(self, state: ChatState) -> Dict[str, Any]:
        """Generate a response for the Base chat model.

        – Guarantees the "helpful chatbot" system prompt exists exactly once per
          thread.
        – Injects any uploaded file contents as extra context (base-only).
        """
        # Make a working copy so we do not mutate the original list inside state
        prompt_messages: List[BaseMessage] = list(state["messages"])
        new_state_messages: List[BaseMessage] = []

        # 1. Add the system prompt once
        if not any(isinstance(m, SystemMessage) for m in prompt_messages):
            sys_msg = SystemMessage(content="You are a helpful and friendly chatbot.")
            prompt_messages.insert(0, sys_msg)
            new_state_messages.append(sys_msg)  # persist system msg in thread state

        # 2. If the caller supplied file content, prepend it as context (only for this round)
        file_chunks = state.get("file_contents", [])
        if file_chunks:
            ctx_msg = SystemMessage(content="Additional context provided by the user:\n\n" + "\n\n".join(file_chunks))
            prompt_messages.insert(1, ctx_msg)
            # Do *not* add ctx_msg to new_state_messages; we don't want to store bulky files forever

        # 3. Invoke the underlying LLM
        if self.use_kv_cache:
            prompt_string = self.llm.tokenizer.apply_chat_template(
                prompt_messages, add_generation_prompt=True, tokenize=False
            )
            response_text = self.llm.invoke(prompt_string)
        else:
            response_text = self.llm.invoke(prompt_messages).content

        # 4. Clean up any special tokens (e.g., Llama chat template tags)
        cleaned_response = self._strip_special_tokens(response_text)

        new_state_messages.append(AIMessage(content=cleaned_response))
        return {"messages": new_state_messages}

    @staticmethod
    def _strip_special_tokens(text: str) -> str:
        """Remove Llama chat template tags like <|...|>."""
        import re
        # Keep content after final assistant tag if present
        if "<|assistant" in text or "assistant<|end_header_id|>" in text:
            # split on the assistant header if present and take the last part
            text = re.split(r"<\|start_header_id\|>assistant<\|end_header_id\|>", text)[-1]
        # Remove all residual <|...|> patterns
        text = re.sub(r"<\|[^>]+\|>", "", text)
        return text.strip()

class LangChainRAG(LangChainBase):
    """A LangChain implementation with Retrieval Augmented Generation (RAG) using LangGraph."""

    def __init__(self, model_params: ModelParameters, **kwargs):
        super().__init__(model_params=model_params, **kwargs)

        model_version = model_params["model_version"]
        if model_version != "RAG":
            raise ValueError(f"Invalid model_version for LangChainRAG: '{model_version}'. Must be 'RAG'.")
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts([""], self.embeddings) 
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        self.history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        
        self.document_chain_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="messages"), 
            ("user", "{input}"),
        ])

    def _add_documents_node(self, state: ChatState) -> Dict[str, Any]:
        if state["file_contents"]:
            logging.info("Adding {} new file(s) to retriever.".format(len(state['file_contents'])))
            combined_content = "\n\n".join(state["file_contents"])
            texts = self.text_splitter.split_text(combined_content)
            docs = [Document(page_content=t) for t in texts]
            
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vectorstore.add_documents(docs)
            
            logging.info(f"Successfully added {len(docs)} document chunks to vector store.")
            return {"rag_enabled": True} 
        return {}

    def _route_to_rag_or_base(self, state: ChatState) -> str:
        if state.get("rag_enabled", False) and self.vectorstore.index.ntotal > 0:
            logging.info("Routing to RAG path.")
            return "retrieve"
        else:
            logging.info("Routing to base chat model path.")
            return "base_chat"

    def _retrieve_node(self, state: ChatState) -> Dict[str, Any]:
        question = state["messages"][-1].content 
        chat_history = state["messages"][:-1] 

        retriever = self.vectorstore.as_retriever()

        history_aware_retriever_chain = create_history_aware_retriever(self.llm, retriever, self.history_aware_retriever_prompt)
        
        retrieved_docs = history_aware_retriever_chain.invoke({
            "input": question,
            "messages": chat_history 
        })
        logging.info(f"Retrieved {len(retrieved_docs)} documents.")
        return {"context": retrieved_docs, "messages": state["messages"]}

    def _generate_response_node(self, state: ChatState) -> Dict[str, Any]:
        context_docs = state["context"]
        question = state["messages"][-1].content
        chat_history = state["messages"][:-1]

        document_chain = create_stuff_documents_chain(self.llm, self.document_chain_prompt)

        response = document_chain.invoke({
            "context": context_docs,
            "input": question,
            "messages": chat_history
        })

        return {"messages": [AIMessage(content=response)]}

# --- Global LangGraph App Getters ---

def get_global_base_app(model_params: Optional[ModelParameters] = None) -> Any:
    """Returns the single, globally compiled LangGraph application for the Base model."""
    global _global_base_app
    if _global_base_app is None:
        logging.info("Initializing global Base LangGraph application.")
        # Default model parameters for the global base app
        if model_params is None:
            model_params = {
                "framework_type": "LangChain",
                "backend": "HuggingFace",
                "model_version": "Base",
                "hf_params": {
                    "model_name": "meta-llama/Llama-3.1-8B-Instruct", # Default to 8B with KV cache
                    "use_kv_cache": True 
                }
            }
        
        base_llm_instance = LangChainBase(model_params=model_params)

        workflow = StateGraph(ChatState)
        workflow.add_node("call_model", base_llm_instance._call_model_node)
        workflow.set_entry_point("call_model")
        workflow.add_edge("call_model", END)
        _global_base_app = workflow.compile(checkpointer=MemorySaver())
        logging.info("Global Base LangGraph application initialized.")
    return _global_base_app

def set_global_app(model_params: ModelParameters) -> None:
    """Rebuild the requested global LangGraph application with new parameters.

    This is used by ChatManager.change_global_model to hot-swap the underlying
    LLM without restarting the server.
    """
    global _global_base_app, _global_rag_app

    version = model_params.get("model_version", "Base")
    if version == "RAG":
        _global_rag_app = None  # Drop existing compiled graph (will be GC'd)
        _global_rag_app = get_global_rag_app(model_params)
        _global_base_app = None  # Ensure only one active global
    else:
        _global_base_app = None
        _global_base_app = get_global_base_app(model_params)
        _global_rag_app = None


def get_global_rag_app(model_params: Optional[ModelParameters] = None) -> Any:
    """Returns the single, globally compiled LangGraph application for the RAG model."""
    global _global_rag_app
    if _global_rag_app is None:
        logging.info("Initializing global RAG LangGraph application.")
        # Default model parameters for the global RAG app
        if model_params is None:
            model_params = {
                "framework_type": "LangChain",
                "backend": "HuggingFace",
                "model_version": "RAG",
                "hf_params": {
                    "model_name": "meta-llama/Llama-3.1-8B-Instruct", # Default to 8B with KV cache for RAG also
                    "use_kv_cache": True 
                }
            }
        
        rag_llm_instance = LangChainRAG(model_params=model_params)

        workflow = StateGraph(ChatState)
        workflow.add_node("add_documents", rag_llm_instance._add_documents_node)
        workflow.add_node("retrieve", rag_llm_instance._retrieve_node)
        workflow.add_node("generate_response", rag_llm_instance._generate_response_node)
        workflow.add_node("base_chat_model", rag_llm_instance._call_model_node) # Can still use base chat logic in RAG

        workflow.set_entry_point("add_documents")
        workflow.add_conditional_edges(
            "add_documents",
            rag_llm_instance._route_to_rag_or_base,
            {
                "retrieve": "retrieve",
                "base_chat": "base_chat_model"
            }
        )
        workflow.add_edge("retrieve", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("base_chat_model", END) # End of path if no RAG needed

        _global_rag_app = workflow.compile(checkpointer=MemorySaver())
        logging.info("Global RAG LangGraph application initialized.")
    return _global_rag_app

# --- Entry point for initial app setup (optional, can be done via http_api) ---
# When the module is first imported, we might want to initialize the default base app
# This is mainly for convenience and ensures an app is ready if no specific model parameters are provided
# get_global_base_app() # Commented out: better to lazy load when first requested via API
