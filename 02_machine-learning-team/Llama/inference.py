import socket
import json


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

    """
    Acts as the interface for chatting with RoboDoc.

        Params:
            user_input (str): The current input provided by the user.
            chat_history (list): A list of dictionaries representing the chat history.
            nodes_from_subgraph (list, optional): A list of Nodes from relevant PrimeKG subgraphs. Defaults to None.
            image_captioning (str, optional): A description of an image provided by the CVTeam. Defaults to None.

        Returns:
            tuple: Contains the user input, model response, and updated chat history.
    """
    return send_prompt(user_input, chat_history, nodes_from_subgraph, image_captioning)