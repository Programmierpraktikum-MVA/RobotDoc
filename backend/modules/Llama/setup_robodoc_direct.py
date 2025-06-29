import json
import torch
import os
import socket
import torch
from unsloth import FastLanguageModel

model_name = "KennyDain/Llama3_RoboDoc_4bit_bnb"

# Global variables to track if model is loaded
model = None
tokenizer = None

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    # change the padding tokenizer value
    #tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    #model.config.pad_token_id = tokenizer.pad_token_id # updating model config
    #tokenizer.padding_side = 'right' # padding to right (otherwise SFTTrainer shows warning)

    FastLanguageModel.for_inference(model)

    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {str(e)}")

def brechstange(llama_out, cutpoints):
    start_idx = -1
    for cutpoint in cutpoints:
        idx = llama_out.find(cutpoint)
        if idx != -1:
            if start_idx == -1 or idx < start_idx:
                start_idx = idx
    if start_idx != -1:
        return llama_out[:start_idx]
    else:
        return llama_out

#cut string for the relevant part:
def cut_string(llama_out, user_input):
  cutpoint = user_input
  start_idx = llama_out.find(cutpoint)
  if start_idx != -1:
    return llama_out[start_idx + len(cutpoint):]
  else:
    return llama_out

def chat_with_robodoc(user_input, chat_history=None, nodes_from_subgraph=None, image_captioning=None):
    global model, tokenizer
    # Initialize chat_history if None
    if chat_history is None:
        chat_history = []

    # Construct instruction based on chat history, nodes, and image captioning
    instruction = "Answer the users question regarding this Patient, if you cant suggest something try to understand the patients situation"
    if len(chat_history) > 0:
        formatted_history = [f"{msg['role']}:{msg['content']}" for msg in chat_history if msg['role'] != 'system']
        instruction += "\n".join(formatted_history)
    if nodes_from_subgraph:
        instruction += f"\n\nNode Info:\n{nodes_from_subgraph}"
    if image_captioning:
        instruction += f"\n\nThe user has also uploaded a picture to better describe his symptoms. Described picture:\n{image_captioning}"

    try:
        # Define the prompt template
        alpaca_prompt = """You are an AI assistant supporting a doctor. Limit your response to 100 character.

        ### Instruction:
        {}

        ### Input:
        {}

        ### Response:
        {}"""

        # Format the input prompt
        prompt_text = alpaca_prompt.format(instruction, user_input, "")

        #print("Prompt Text: ")
       # print(prompt_text)
        # Tokenize inputs
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        #print("tokenized/inputs: ")
       # print(inputs)

        # Generate output
        outputs = model.generate(
            **inputs,  # Use ** to unpack the dictionary
            max_new_tokens=200,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,  # Ensure padding token is set to EOS token ID
           # eos_token_id=tokenizer.eos_token_id   # Specify EOS token ID for proper handling
        )

        #test_decode = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        #print("tokenizer.batch_decode: ")
        #print(test_decode)

        # Decode the output tokens
        model_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print("model_response before ###ResponseSplit: ")
        #print(model_response)


        #test_decode = tokenizer.batch_decode(outputs, skip_special_tokens = True)
        #print("tokenizer.batch_decode: ")
        #print(model_response)


        # Extract only the response part (excluding prompt and instructions)
        model_response = model_response.split("### Response:")[1].strip()
        #print("model_response2: ")
        #print(model_response)
        cutpoints = [" You are an AI assistant", "### Instruction:", "### Input:"]
        model_response = brechstange(model_response, cutpoints)


        
        #response_split = model_response.split("### Response:")
        #if len(response_split) > 1:
        #    model_response = response_split[1].strip()
        #else:
        #    model_response = model_response.strip()
     
        # Update chat_history with user_input and model_response
        chat_history.append(f"user:{user_input}")
        chat_history.append(f"assistant:{model_response}")

        print("chatHistory: ")
        print(chat_history)

        print("ModelResponse: ")
        print(model_response)
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
                    print("received_data: ")
                    print(received_data)
                    user_input = received_data['user_input']
                    chat_history = received_data['chat_history']
                    nodes_from_subgraph = received_data.get('nodes_from_subgraph', None)
                    image_captioning = received_data.get('image_captioning', None)
                    print("received_data: ")
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
  listen_for_prompts()
