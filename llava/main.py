import os
import json
import socket
import torch
from PIL import Image
from transformers import LlavaNextProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration

# Model and quantization configuration
MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
IMAGE_DIR = "img"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 65533

def load_image(image_file):
    """Load an image and convert to RGB."""
    return Image.open(image_file).convert("RGB")

def list_to_string(items):
    """Convert a list to a space-separated string."""
    if not isinstance(items, list):
        raise ValueError("Input has to be a list.")
    return ' '.join(str(item) for item in items)

def cut_string(text, cutpoint="ASSISTANT: "):
    """Extract substring after cutpoint."""
    start_idx = text.find(cutpoint)
    return text[start_idx + len(cutpoint):] if start_idx != -1 else ""

def process_image(processor, model, image_name):
    """Process image and generate model response."""
    try:
        image_path = os.path.join(IMAGE_DIR, image_name)
        image = load_image(image_path)
        messages = [{
            "role": "user",
            "content": "<image> What symptoms can the person recognize?"
        }]
        inputs = processor(messages, images=image, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=384)
            output = processor.batch_decode(generated, skip_special_tokens=True)
        generated_string = list_to_string(output)
        return cut_string(generated_string)
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error processing image."

def handle_client(conn, processor, model):
    """Handle a single client connection."""
    with conn:
        print('Connected by', conn.getpeername())
        while True:
            data = conn.recv(4096)
            if not data:
                break
            try:
                received_data = json.loads(data.decode('utf-8'))
                image_name = received_data.get('image_file')
                model_response = process_image(processor, model, image_name)
                response_data = {"model_response": model_response}
            except Exception as e:
                print(f"Error decoding JSON: {e}")
                response_data = {"model_response": "Error decoding JSON."}
            conn.sendall(json.dumps(response_data).encode('utf-8'))

def listen_for_prompts():
    """Load model and listen for incoming prompts."""
    print("Loading model into VRAM...")
    try:
        processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            quantization_config=QUANT_CONFIG,
            device_map="auto"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((SERVER_HOST, SERVER_PORT))
        server_socket.listen()
        print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}...")
        while True:
            conn, _ = server_socket.accept()
            handle_client(conn, processor, model)

if __name__ == "__main__":
    listen_for_prompts()
