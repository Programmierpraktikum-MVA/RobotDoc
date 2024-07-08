import socket
import json
import io
from PIL import Image

def load_image_from_bytes(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

def send_prompt(image_file):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('127.0.0.1', 65533))
        data = {
            "image_file": image_file
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
