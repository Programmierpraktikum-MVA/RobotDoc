from inference import chat_with_robodoc

user_input = "The Patient has a mild fever."
chat_history = []
nodes_from_subgraph = "The Content: angioedema, disease, The Content: acquired angioedema, disease"
image_captioning = "Male with brown hair." 

     # Invoke chat_with_robodoc with example data
response_data = chat_with_robodoc(user_input, chat_history, nodes_from_subgraph, image_captioning)

print("User Input:", response_data["user_input"])
print("Model Response:", response_data["model_response"])
print("Updated History:")
for message in response_data["chat_history"]:
    print(f"{message['role'].capitalize()}: {message['content']}")
