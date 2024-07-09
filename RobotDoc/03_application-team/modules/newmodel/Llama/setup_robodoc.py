import json
import torch
import os
import socket
import torch
from unsloth import FastLanguageModel

model_name = "KennyDain/Llama3_medQA_bnb_4bit"
save_directory = "./model"

# Global variables to track if model is loaded
model_loaded = False
model = None
tokenizer = None


def download_and_save_model(model_name, save_directory):
    global model, tokenizer
    try:
        
        if os.path.exists(save_directory):
            print("Model already downloaded")
            return True

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        # Save tokenizer and model to disk
        tokenizer.save_pretrained(save_directory)
        model.save_pretrained(save_directory)
        print(f"Model and tokenizer saved to {save_directory}")
        #unload_model()
        return True

    except Exception as e:
        print(f"Failed to download and save model: {str(e)}")
        return False


def load_model():
    global model, model_loaded, tokenizer
    if not model_loaded:
        print("Loading model into VRAM...")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=save_directory,
                max_seq_length=2048,
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )

            
            model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")


def unload_model():
    global model, model_loaded
    if model_loaded:
        del model
        torch.cuda.empty_cache()
        model_loaded = False
        print("Model unloaded from VRAM.")

def cut_string(llama_out):
    cutpoint = "You are an AI assistant"
    start_idx = llama_out.find(cutpoint)
    if (start_idx != -1):
        return llama_out[:start_idx]
    else:
        return llama_out

def cut_string_hash(llama_out):
    cutpoint = "### Instruction:"
    start_idx = llama_out.find(cutpoint)
    if (start_idx != -1):
        return llama_out[:start_idx]
    else:
        return llama_out


def chat_with_robodoc(user_input, chat_history=None, nodes_from_subgraph=None, image_captioning=None):
    global model_loaded, model,tokenizer

    print("reached function")

    if chat_history is None:
        chat_history = []
    


    print(chat_history)    

    instruction = ""
    if len(chat_history) > 0:
        instruction += f"\n\nChatHistory: \n{chat_history}"
    if nodes_from_subgraph:
        instruction += f"\n\nNode Info:\n{nodes_from_subgraph}"
    if image_captioning:
        instruction += f"\n\nThe user has also uploaded a picture to better describe his symptoms. Described picture:\n{image_captioning}"


    if(model_loaded):
        print("modelloaded, skipping")
        load_model()
    else:
        print("Loading model")
        load_model()
        
    
        
    try:

        alpaca_prompt = """

        You are an AI assistant supporting a doctor. The user describes patient symptoms. If the Doctor chooses to you receive node names and node types from a knowledge graph for further reference. These are not confirmed diseases or drugs. Limit responses to 200 characters. For suspected diseases, ask for specific details.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        prompt_text = alpaca_prompt.format(instruction, user_input, "")

        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id, # Ensure padding token is set to EOS token ID
            eos_token_id=tokenizer.eos_token_id  # Specify EOS token ID for proper handling
        )


        # Decode the output tokens
        model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response part (excluding prompt and instructions)
        model_response = model_response.split("### Response:")[1].strip()
    
        # model_response = cut_string(model_response)
        # model_response = cut_string_hash(model_response)

        # Update chat_history with user_input and model_response
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "model_response", "content": model_response})

        print("Modelresponse:")
        print(model_response)

        print("chat History:")
        print(chat_history)    
            
        # unload_model()
       
        # Return user_input, model_response, and updated chat_history
        return user_input, model_response, chat_history

    except Exception as e:
        print(f"Exception occurred during model generation: {str(e)}")
        return "Sorry, I couldn't process your request at the moment.", None, chat_history


def listen_for_prompts():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 65143))
        s.listen()
        print("Waiting for connections...")
        while True:  # Keep listening for connections indefinitely
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:  # Keep processing prompts within the connection
                    data = conn.recv(4096)
                    if not data:
                        break
                    received_data = json.loads(data.decode('utf-8'))
                    user_input = received_data['user_input']
                    chat_history = received_data['chat_history']
                    nodes_from_subgraph = received_data.get('nodes_from_subgraph', None)
                    image_captioning = received_data.get('image_captioning', None)
                    user_question, model_response, updated_history = chat_with_robodoc(user_input, chat_history,
                                                                                       nodes_from_subgraph,
                                                                                       image_captioning)
                    response_data = {
                        "user_input": user_question,
                        "model_response": model_response,
                        "chat_history": updated_history
                    }
                    conn.sendall(json.dumps(response_data).encode('utf-8'))


if __name__ == "__main__":
    # Download and save the model if not already saved
    if not download_and_save_model(model_name, save_directory):
        print("Failed to download and save model. Exiting...")
    else:
        # Start listening for prompts
        listen_for_prompts()
