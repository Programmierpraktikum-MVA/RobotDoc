import socket
import json


class LLM:
    def __init__(self):
        self.patients = {}
        self.patInfo = {}

        self.system_message = {
            "role": "robotdoc",
            "content": "You are an AI assistant supporting a doctor. The user describes patient symptoms. You receive node names and node types from a knowledge graph. Limit responses to 200 characters. For suspected diseases, ask for specific details."
        }

    def add_message(self, patient_id, message):
        if patient_id not in self.patients:
            self.patients[patient_id] = []
        self.patients[patient_id].append(message)


    

    def chat_with_robotdoc(self, patient_id, patient_info, user_input, nodes_from_subgraph=None, image_captioning=None):
        patient_info_str = str(patient_info)
        
        chat_history = get_chat_history_for_patient(patient_id)
        
        try:
            model_response = query_llm_model(user_input=user_input, chat_history=chat_history,
                                                         nodes_from_subgraph=nodes_from_subgraph,
                                                         image_captioning=image_captioning)
            response = model_response["model_response"]
        except Exception as e:
            print(f"Exception occurred during model generation: {str(e)}")
            response = "Sorry, I couldn't process your request at the moment."

        self.add_message(patient_id, {"role": "user", "content": user_input})
        self.add_message(patient_id, {"role": "robotdoc", "content": response})
        return user_input, response
    


def send_prompt(user_input, chat_history, nodes_from_subgraph=None, image_captioning=None):
    data = {
        "user_input": user_input,
        "chat_history": chat_history
    }
    if nodes_from_subgraph:
        data["nodes_from_subgraph"] = nodes_from_subgraph
    if image_captioning:
        data["image_captioning"] = image_captioning

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('robotdoc-llama', 65143))
            s.sendall(json.dumps(data).encode('utf-8'))
            response = s.recv(4096)

            return json.loads(response.decode('utf-8'))

    except Exception as e:
        print(f"Exception occurred during socket communication: {str(e)}")
        return {
            "user_input": user_input,
            "model_response": "Sorry, I couldn't connect to the AI model.",
            "chat_history": chat_history
        }


def query_llm_model(user_input, chat_history, nodes_from_subgraph=None, image_captioning=None):
    return send_prompt(user_input, chat_history, nodes_from_subgraph, image_captioning)

def get_chat_history_for_patient(patient_id):
    from util.db_model import ChatMessage  # Local import to avoid circular dependency
    messages = ChatMessage.query.filter_by(patient_id=patient_id).order_by(ChatMessage.timestamp).all()
    return [
        {
            'role': 'user' if m.sender.lower() == 'user' else 'robotdoc',
            'content': m.message
        }
        for m in messages
    ]



