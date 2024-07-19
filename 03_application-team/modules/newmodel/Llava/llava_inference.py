import socket
import json
from PIL import Image
import io
from io import BytesIO

def load_image_from_bytes(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        print(f"error loading img: {e}")
        raise

def transform_image_to_hex(image_file):
    image = Image.open(image_file).convert("RGB")
    byte_io = BytesIO()
    image.save(byte_io, format='PNG')
    image_bytes = byte_io.getvalue()
    image_hex = image_bytes.hex
    return image_hex

def send_prompt(image_file):
    try:
        image_hex = transform_image_to_hex(image_file)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', 65533))
            data = {
                "image_file": image_hex
            }

            s.sendall(json.dumps(data).encode('utf-8'))
            response = s.recv(8192)
            return json.loads(response.decode('utf-8'))
    except Exception as e:
        print(f"Error send prompt: {e}")

def image_captioning_with_robodoc(image_file):

    """
    Acts as the interface for image_captioning with RoboDoc.

        Params:
            user_input: Image bytes

        Returns:
            llava_output (str): Image Caption
    """
    return send_prompt(image_file)
