import socket
import json
from PIL import Image

def send_prompt(user_input):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', 65143))
        data = {
            "user_input": user_input
        }

        s.sendall(json.dumps(data).encode('utf-8'))
        response = s.recv(4096)
        return json.loads(response.decode('utf-8'))


def image_captioning_with_robodoc(image_file):

    """
    Acts as the interface for image_captioning with RoboDoc.

        Params:
            user_input (str): An Image

        Returns:
            tuple: Contains the user input, model response, and updated chat history.
    """
    return send_prompt(image_file)