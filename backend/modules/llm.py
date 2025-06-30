import socket
import json


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

    def chat_with_robodoc(self, patient_id, patient_info, user_input, nodes_from_subgraph=None, image_captioning=None):
        patient_info_str = str(patient_info)
        chat_history = self.get_chat_history(patient_id)

        try:
            model_response = chat_with_robodoc(user_input=user_input, chat_history=chat_history,
                                                         nodes_from_subgraph=nodes_from_subgraph,
                                                         image_captioning=image_captioning)
            response = model_response["model_response"]
        except Exception as e:
            print(f"Exception occurred during model generation: {str(e)}")
            response = "Sorry, I couldn't process your request at the moment."

        self.add_message(patient_id, {"role": "user", "content": user_input})
        self.add_message(patient_id, {"role": "model_response", "content": response})
        return user_input, response
    


def send_prompt(user_input, chat_history, nodes_from_subgraph=None, image_captioning=None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', 65143))
        data = {
            "user_input": user_input,
            "chat_history": chat_history
        }
        if nodes_from_subgraph:
            data["nodes_from_subgraph"] = nodes_from_subgraph
        if image_captioning:
            data["image_captioning"] = image_captioning

        s.sendall(json.dumps(data).encode('utf-8'))
        response = s.recv(4096)
        return json.loads(response.decode('utf-8'))


def chat_with_robodoc(user_input, chat_history, nodes_from_subgraph=None, image_captioning=None):
    return send_prompt(user_input, chat_history, nodes_from_subgraph, image_captioning)

