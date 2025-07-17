# chat_manager.py
# This file manages the model, the tokenizer, AND the conversation history.

import datetime
import random
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# In-memory storage for conversation histories
conversation_histories = {}

# ==============================================================================
# --- Placeholder Function ---
# ==============================================================================
def generate_hardcoded_chat(user_input: str) -> str:
    """Simulates a response using simple keyword matching."""
    lower_input = user_input.lower()
    if "hello" in lower_input or "hi" in lower_input:
        return "Hello there! How can I assist you today?"
    elif "how are you" in lower_input:
        return "I'm just a set of instructions, but I'm running perfectly! Thanks for asking."
    elif "time" in lower_input:
        now = datetime.datetime.now()
        current_time = now.strftime("%I:%M %p")
        return f"The current time is {current_time}."
    else:
        return "I'm not sure how to respond to that."

# ==============================================================================
# --- Core Model Logic ---
# ==============================================================================
model = None
tokenizer = None
model_loaded = False

def load_real_model():
    """Loads the tokenizer and a pre-trained model into memory."""
    global model, tokenizer, model_loaded
    if model_loaded:
        return True

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logging.info(f"Found {gpu_count} available GPU(s).")
            for i in range(gpu_count):
                logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logging.warning("No GPU found. The model will run on the CPU.")

        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        logging.info(f"Loading model: {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        # Check pytorch params
        print(type(next(model.parameters())))

        logging.info(f"Model successfully loaded on device: {model.device}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        model_loaded = True
        return True
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        return False

def _generate_response_with_history(history: list) -> str:
    """Internal function to generate a response from a given history."""
    if not model_loaded:
        return "The AI model is not loaded."
    try:
        input_ids = tokenizer.apply_chat_template(
            history,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        output_sequences = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        response = output_sequences[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True)
    except Exception as e:
        logging.error(f"An error occurred during model generation: {e}")
        return "An error occurred during model generation."

# --- UPDATED: Function now accepts optional file_content ---
def invoke_chat_model(user_input: str, session_id: str, file_content: str = None) -> str:
    """
    Manages a conversation session, including history and optional file context.
    """
    history = conversation_histories.get(session_id, [])
    
    # --- UPDATED: If history is new, create a system prompt with context ---
    if not history:
        if file_content:
            # Create a detailed system prompt including the file content
            system_prompt = f"""You are a helpful and friendly chatbot. You will answer questions based on the provided document.
---
DOCUMENT CONTENT:
{file_content}
---
Now, please answer the user's question."""
        else:
            # Use the default system prompt
            system_prompt = "You are a helpful and friendly chatbot."
        
        history.append({"role": "system", "content": system_prompt})

    history.append({"role": "user", "content": user_input.strip()})
    bot_response_text = _generate_response_with_history(history)
    history.append({"role": "assistant", "content": bot_response_text})
    conversation_histories[session_id] = history

    return bot_response_text
