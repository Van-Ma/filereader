# models/langchain.py
# This file contains the stateful, KV-caching chat implementation.

import logging
from typing import Any, Optional, List, Dict, Sequence, Union, Tuple
import torch

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.chat_models import ChatHuggingFace # Import ChatHuggingFace

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict, Annotated

from models.custom_huggingface_llm import CustomHuggingFaceLLM, HuggingFaceParameters # Import HuggingFaceParameters from its new location
from models.model_parameters import ModelParameters # Only import ModelParameters from chat_manager

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

class LangChainBase:
    """A base LangChain implementation using LangGraph for state management and a custom LLM."""
    
    def __init__(self, model_params: ModelParameters, **kwargs):
        # Decentralized Validation for LangChainBase
        framework_type = model_params["framework_type"]
        backend = model_params["backend"]
        model_version = model_params["model_version"]
        hf_params = model_params.get("hf_params") # Get hf_params, will be validated below

        if framework_type != "LangChain":
            raise ValueError(f"Invalid framework_type for LangChainBase: '{framework_type}'. Must be 'LangChain'.")
        if backend != "HuggingFace":
            raise ValueError(f"Invalid backend for LangChainBase: '{backend}'. Must be 'HuggingFace' for now.")
        if model_version != "Base":
            raise ValueError(f"Invalid model_version for LangChainBase: '{model_version}'. Must be 'Base'.")
        
        # Validate hf_params presence and type for HuggingFace backend
        if not hf_params or not isinstance(hf_params, dict):
            raise ValueError("Missing or invalid 'hf_params' for HuggingFace backend in LangChainBase.")
        # Convert to TypedDict for type safety and access (already validated structurally in chat_manager)
        hf_params_typed: HuggingFaceParameters = hf_params 

        if hf_params_typed["model_name"] not in ALLOWED_HUGGINGFACE_MODELS:
            raise ValueError("Invalid model_name for LangChainBase: '{}'. Must be one of {}".format(hf_params_typed['model_name'], list(ALLOWED_HUGGINGFACE_MODELS)))

        self.model_id = hf_params_typed["model_name"]
        self.use_kv_cache = hf_params_typed["use_kv_cache"]

        if self.use_kv_cache:
            self.llm = CustomHuggingFaceLLM(hf_params=hf_params_typed)
        else:
            logging.info("Initializing ChatHuggingFace with HuggingFacePipeline model '{}' (no explicit KV cache across invocations).".format(self.model_id))
            try:
                # Use ChatHuggingFace to wrap HuggingFacePipeline for tutorial-like message passing
                pipeline_llm = HuggingFacePipeline.from_model_id(
                    model_id=self.model_id,
                    task="text-generation",
                    device=0 if torch.cuda.is_available() else -1, # -1 for CPU, 0 for first GPU
                    pipeline_kwargs={
                        "max_new_tokens": 256, 
                        "do_sample": True, 
                        "temperature": 0.6, 
                        "top_p": 0.9,
                    }
                )
                self.llm = ChatHuggingFace(llm=pipeline_llm) # Wrap the pipeline with ChatHuggingFace
            except Exception as e:
                logging.error("Failed to load HuggingFacePipeline model '{}'. This could be due to: ".format(self.model_id) +
                              "1. Invalid model name."
                              "2. Insufficient memory/GPU resources."
                              "3. Network issues preventing model download."
                              "4. Hugging Face rate limits or authentication issues (for private models)."
                              "Error details: {}".format(e))
                raise # Re-raise the exception to propagate the error

        self.checkpointer = MemorySaver()

        self.base_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly chatbot."),
            MessagesPlaceholder(variable_name="messages")
        ])

        workflow = StateGraph(ChatState)
        workflow.add_node("call_model", self._call_model_node)
        workflow.set_entry_point("call_model")
        workflow.add_edge("call_model", END)
        self.app = workflow.compile(checkpointer=self.checkpointer)

    def _call_model_node(self, state: ChatState) -> Dict[str, Any]:
        messages = state["messages"]

        response_text = ""
        if self.use_kv_cache:
            # For CustomHuggingFaceLLM with KV cache, explicitly use tokenizer to format
            if hasattr(self.llm, 'tokenizer') and self.llm.tokenizer:
                prompt_string = self.llm.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                response_text, _ = self.llm.invoke_with_cache(prompt_string)
            else:
                # Fallback if tokenizer is unexpectedly missing (should not happen for CustomHuggingFaceLLM)
                logging.error("CustomHuggingFaceLLM instance missing tokenizer. Cannot format prompt.")
                raise ValueError("LLM tokenizer missing for KV cache model.")
        else:
            # For ChatHuggingFace (no explicit KV cache), pass messages directly to invoke
            response_text = self.llm.invoke(messages).content # .content to get string from AIMessage
        

        return {"messages": [AIMessage(content=response_text)]}

    def invoke(self, session_id: str, user_input: str, file_content: Optional[Union[str, List[str]]] = None) -> str:
        config = {"configurable": {"thread_id": session_id}}
        
        input_messages = [HumanMessage(content=user_input)]
        graph_input: ChatState = {
            "messages": input_messages,
            "file_contents": [], # Ensure this is always initialized
            "rag_enabled": False # Ensure this is always initialized
        }
        
        if file_content:
            if isinstance(file_content, str):
                graph_input["file_contents"].append(file_content)
            elif isinstance(file_content, list):
                graph_input["file_contents"].extend(file_content)
            else:
                logging.warning("Unsupported file_content type: {}. Ignoring.".format(type(file_content)))

        output = self.app.invoke(graph_input, config)
        
        ai_message = output["messages"][-1]
        return ai_message.content

    def get_context_history(self, session_id: str) -> List[Dict[str, str]]:
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = self.app.get_state(config)
            history_dicts = []
            for message in state.messages:
                history_dicts.append({"role": message.type, "content": message.content})
            return history_dicts
        except Exception as e:
            logging.warning(f"Could not retrieve state for session {session_id}: {e}")
            return []

    def clear_context(self, session_id: str):
        config = {"configurable": {"thread_id": session_id}}
        try:
            self.app.clear(config)
            if self.use_kv_cache and hasattr(self.llm, 'reset_kv_cache'):
                self.llm.reset_kv_cache()
            logging.info(f"Cleared context and KV cache (if applicable) for session {session_id}.")
        except Exception as e:
            logging.error(f"Failed to clear context for session {session_id}: {e}")


class LangChainRAG(LangChainBase):
    """A LangChain implementation with Retrieval Augmented Generation (RAG) using LangGraph."""

    def __init__(self, model_params: ModelParameters, **kwargs):
        super().__init__(model_params=model_params, **kwargs)

        # Decentralized Validation for LangChainRAG (additional checks or overrides)
        model_version = model_params["model_version"]
        if model_version != "RAG":
            raise ValueError(f"Invalid model_version for LangChainRAG: '{model_version}'. Must be 'RAG'.")

        # It inherits agent_framework, backend, model_name, and options validation from LangChainBase's super().__init__
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_texts([""], self.embeddings) 
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        workflow = StateGraph(ChatState)
        workflow.add_node("add_documents", self._add_documents_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("base_chat_model", self._call_model_node)

        workflow.set_entry_point("add_documents")
        workflow.add_conditional_edges(
            "add_documents",
            self._route_to_rag_or_base,
            {
                "retrieve": "retrieve",
                "base_chat": "base_chat_model"
            }
        )

        workflow.add_edge("retrieve", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("base_chat_model", END)

        self.app = workflow.compile(checkpointer=self.checkpointer)

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

    def invoke(self, session_id: str, user_input: str, file_content: Optional[Union[str, List[str]]] = None) -> str:
        config = {"configurable": {"thread_id": session_id}}
        
        input_messages = [HumanMessage(content=user_input)]
        graph_input: ChatState = {
            "messages": input_messages,
            "file_contents": [], 
            "rag_enabled": False 
        }

        if file_content:
            if isinstance(file_content, str):
                graph_input["file_contents"].append(file_content)
            elif isinstance(file_content, list):
                graph_input["file_contents"].extend(file_content)
            else:
                logging.warning(f"Unsupported file_content type in LangChainRAG invoke: {type(file_content)}. Ignoring.")

        output = self.app.invoke(graph_input, config)
        
        ai_message = output["messages"][-1]
        return ai_message.content
