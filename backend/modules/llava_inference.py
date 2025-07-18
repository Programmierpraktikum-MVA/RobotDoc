import socket
import json
from PIL import Image
import io

def load_image_from_bytes(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"error loading img: {e}")
        raise

def send_prompt(image_name):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('robotdoc-llava', 65533))
            data = {
                "image_file": image_name
            }

            s.sendall(json.dumps(data).encode('utf-8'))
            response = s.recv(4096)
            return json.loads(response.decode('utf-8'))
    except Exception as e:
        print(f"Error send prompt: {e}")

def image_captioning_with_robodoc(image_name):

    """
    Acts as the interface for image_captioning with RoboDoc.

        Params:
            user_input (str): Image Name

        Returns:
            llava_output (str): Image Caption
    """
    return send_prompt(image_name)
