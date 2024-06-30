
# import torch
# import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import pickle


# # Load LLM model directly

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
# )

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", quantization_config=bnb_config, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     model_kwargs={"torch_dtype": torch.bfloat16},
# )

# # Load the graph object
# graph = pickle.load(open('datasets/graph_22.p','rb'))

# def extract_content(graph):

#   prompts_with_context = []

#   nodes = list(graph.nodes(data=True))

#   for node in nodes:
#       name = node[1]['name']
#       type = node[1]['type']
#       context_prompt = f"The Content: {name}, {type}"
#       prompts_with_context.append(context_prompt)

#   return prompts_with_context


# # Process the graph using the function
# content = extract_content(graph)
# print(content)

# nodes_from_subgraph = "\n".join(content)

# messages = [
#     {
#         "role":"system",
#         "content":"You are an AI assistant supporting a doctor. The user describes patient symptoms. You receive node names and node types from a knowledge graph. Limit responses to 200 characters. For suspected diseases, ask for specific details."+content_str
#     },
#     {
#         "role":"user",
#         "content":"John Doe has a history of hypertension, which is well-controlled with medication. He has no history of allergies or surgeries. He is not currently taking any medication except for his blood pressure medication"
#     }
#   ]

# prompt = pipeline.tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     )

# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#   ]


# outputs = pipeline(
#     prompt,
#     max_new_tokens = 256,
#     eos_token_id = terminators,
#     do_sample = True,
#     temperature = 0.6,
#     top_p = 0.9,
#     )
# print(outputs[0]["generated_text"][len(prompt):])

# system_message = {
#       "role": "system",
#       "content": "You are an AI assistant supporting a doctor. The user describes patient symptoms. You receive node names and node types from a knowledge graph. Limit responses to 200 characters. For suspected diseases, ask for specific details."
#   }

# chat_history = [system_message]

# def chat_with_robodoc(user_input, nodes_from_subgraph=None):

#     if nodes_from_subgraph:
#         system_message_with_node_info = {
#             "role": "system",
#             "content": f"{system_message['content']}\n\nNode Info:\n{nodes_from_subgraph}"
#         }
#     else:
#         system_message_with_node_info = system_message

#       # Append user input to chat history
#     chat_history.append({"role": "user", "content": user_input})

#       # Construct prompt from updated chat history
#     content_str = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in chat_history])
#     messages = [
#         {"role": "system", "content": system_message_with_node_info["content"] + "\n" + content_str},
#         {"role": "user", "content": user_input}
#     ]

#     prompt = pipeline.tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     try:
#         outputs = pipeline(
#             prompt,
#             max_new_tokens=256,
#             eos_token_id=terminators,
#             do_sample=True,
#             temperature=0.6,
#             top_p=0.9,
#         )

#           # Extract generated response
#         model_response = outputs[0]["generated_text"][len(prompt):].strip()

#     except Exception as e:
#         print(f"Exception occurred during model generation: {str(e)}")
#         model_response = "Sorry, I couldn't process your request at the moment."

#       # Append model response to chat history
#     chat_history.append({"role": "assistant", "content": model_response})

#     return user_input, model_response

# # Example usage with nodes_from_subgraph as a string
# nodes_from_subgraph = "\n".join(content)
# user_input_1 = "John Doe has a history of hypertension, which is well-controlled with medication. He has no history of allergies or surgeries. He is not currently taking any medication except for his blood pressure medication."
# question, answer = chat_with_robodoc(user_input_1, nodes_from_subgraph)
# print(f"Q: {question}")
# print(f"A: {answer}")

import openai
import pickle
import os

# Set your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Load the graph object
# graph = pickle.load(open('util/datasets/graph_01.p', 'rb'))

# def extract_content(graph):
#     prompts_with_context = []

#     nodes = list(graph.nodes(data=True))

#     for node in nodes:
#         name = node[1]['name']
#         type = node[1]['type']
#         context_prompt = f"The Content: {name}, {type}"
#         prompts_with_context.append(context_prompt)

#     return prompts_with_context

# # Process the graph using the function
# content = extract_content(graph)
# print(content)

# nodes_from_subgraph = "\n".join(content)

def chat_with_openai(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response['choices'][0]['message']['content']

system_message = {
    "role": "system",
    "content": "You are an AI assistant supporting a doctor. The user describes patient symptoms. You receive node names and node types from a knowledge graph. Limit responses to 200 characters. For suspected diseases, ask for specific details."
}

chat_history = [system_message]

def chat_with_robodoc(user_input, nodes_from_subgraph=None):
    if nodes_from_subgraph:
        system_message_with_node_info = {
            "role": "system",
            "content": f"{system_message['content']}\n\nNode Info:\n{nodes_from_subgraph}"
        }
    else:
        system_message_with_node_info = system_message

    # Append user input to chat history
    chat_history.append({"role": "user", "content": user_input})

    # Construct prompt from updated chat history
    messages = [
        system_message_with_node_info
    ] + chat_history

    try:
        model_response = chat_with_openai(messages)
    except Exception as e:
        print(f"Exception occurred during model generation: {str(e)}")
        model_response = "Sorry, I couldn't process your request at the moment."

    # Append model response to chat history
    chat_history.append({"role": "assistant", "content": model_response})
    print(chat_history)

    return user_input, model_response

# # Example usage with nodes_from_subgraph as a string
# nodes_from_subgraph = "\n".join(content)
# user_input_1 = "John Doe has a history of hypertension, which is well-controlled with medication. He has no history of allergies or surgeries. He is not currently taking any medication except for his blood pressure medication."
# question, answer = chat_with_robodoc(user_input_1, nodes_from_subgraph)
# print(f"Q: {question}")
# print(f"A: {answer}")
