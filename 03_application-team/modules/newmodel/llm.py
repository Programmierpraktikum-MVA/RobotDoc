
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pickle
import traceback
from modules.newmodel.Llama import inference

class LLM:
    def __init__(self):
        self.patients = {}
        self.patInfo = {}

        self.system_message = {
            "role": "system",
            "content": "You are an AI assistant supporting a doctor. The user describes patient symptoms. You receive node names and node types from a knowledge graph. Limit responses to 200 characters. For suspected diseases, ask for specific details."
        }



    def add_message(self, patient_id, message):
        if patient_id not in self.patients:
            self.patients[patient_id] = []
        self.patients[patient_id].append(message)

    
    def get_chat_history(self, patient_id):
        return self.patients.get(patient_id, [])
    

  

    def chat_with_robodoc(self, patient_id,patient_info, user_input, nodes_from_subgraph=None, image_captioning=None):

   

        

        #patient_info = self.get_patient_info(patient_id)
        patient_info_str = str(patient_info)

        chat_history = self.get_chat_history(patient_id)
            
        messages = [{"role": "user", "content": patient_info_str}] + chat_history


        # content_str = "\n".join([f"{message['role'].capitalize()}: {message['content']}" for message in self.chat_history])
        # messages = [
        #     {"role": "system", "content": system_message_with_node_info["content"] + "\n" + content_str},
        #     {"role": "user", "content": user_input}
        # ]

        try:
            model_response = inference.chat_with_robodoc(user_input=user_input, chat_history=messages, nodes_from_subgraph=nodes_from_subgraph, image_captioning=image_captioning)
            response = model_response["model_response"]
        except Exception as e:
            print(f"Exception occurred during model generation: {str(e)}")
            response = "Sorry, I couldn't process your request at the moment."
        
        self.add_message(patient_id, {"role": "user", "content": user_input})
        self.add_message(patient_id, {"role": "assistant", "content": response})
        return user_input, response

            





# # Example usage
# llm = LLM()
# nodes_from_subgraph = "\n".join(llm.content)
# user_input_1 = "John Doe has a history of hypertension, which is well-controlled with medication. He has no history of allergies or surgeries. He is not currently taking any medication except for his blood pressure medication."
# question, answer = llm.chat_with_robodoc(user_input_1, nodes_from_subgraph)
# print(f"Q: {question}")
# print(f"A: {answer}")


# #############################################################
# import openai
# import pickle
# import os

# # Set your OpenAI API key
# openai.api_key = os.environ.get('OPENAI_API_KEY')

# # Load the graph object
# # graph = pickle.load(open('util/datasets/graph_01.p', 'rb'))

# # def extract_content(graph):
# #     prompts_with_context = []

# #     nodes = list(graph.nodes(data=True))

# #     for node in nodes:
# #         name = node[1]['name']
# #         type = node[1]['type']
# #         context_prompt = f"The Content: {name}, {type}"
# #         prompts_with_context.append(context_prompt)

# #     return prompts_with_context

# # # Process the graph using the function
# # content = extract_content(graph)
# # print(content)

# # nodes_from_subgraph = "\n".join(content)

# def chat_with_openai(messages):
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=messages
#     )
#     return response['choices'][0]['message']['content']

# system_message = {
#     "role": "system",
#     "content": "You are an AI assistant supporting a doctor. The user describes patient symptoms. You receive node names and node types from a knowledge graph. Limit responses to 200 characters. For suspected diseases, ask for specific details."
# }

# chat_history = [system_message]

# def chat_with_robodoc(user_input, nodes_from_subgraph=None):
#     if nodes_from_subgraph:
#         system_message_with_node_info = {
#             "role": "system",
#             "content": f"{system_message['content']}\n\nNode Info:\n{nodes_from_subgraph}"
#         }
#     else:
#         system_message_with_node_info = system_message

#     # Append user input to chat history
#     chat_history.append({"role": "user", "content": user_input})

#     # Construct prompt from updated chat history
#     messages = [
#         system_message_with_node_info
#     ] + chat_history

#     try:
#         model_response = chat_with_openai(messages)
#     except Exception as e:
#         print(f"Exception occurred during model generation: {str(e)}")
#         model_response = "Sorry, I couldn't process your request at the moment."

#     # Append model response to chat history
#     chat_history.append({"role": "assistant", "content": model_response})
#     print(chat_history)

#     return user_input, model_response

# # # Example usage with nodes_from_subgraph as a string
# # nodes_from_subgraph = "\n".join(content)
# # user_input_1 = "John Doe has a history of hypertension, which is well-controlled with medication. He has no history of allergies or surgeries. He is not currently taking any medication except for his blood pressure medication."
# # question, answer = chat_with_robodoc(user_input_1, nodes_from_subgraph)
# # print(f"Q: {question}")
# # print(f"A: {answer}")
