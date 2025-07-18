import os
import json
import torch
import socket
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from PIL import Image

model_name = "llava-hf/llava-1.5-7b-hf"
save_directory = "./model"  # Specify your desired save directory

# Define quantization config
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image     

#transform output list into string
def list_to_string(llava_out):
    if not isinstance(llava_out, list):
        raise ValueError("Input hast to be a list.")

    return ' '.join(str(item) for item in llava_out)

def cut_string(llava_out):
    """Extract only the assistant's part of the response."""
    cutpoint = "ASSISTANT: "
    start_idx = llava_out.find(cutpoint)
    if start_idx != -1:
        return llava_out[start_idx + len(cutpoint):]
    return ""

def listen_for_prompts():
    print("Loading model into VRAM...")
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, quantization_config=quantization_config).to("cuda")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return 

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 65533))
        s.listen()
        print("Waiting for connections...")
        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:
                    data = conn.recv(4096)
                    if not data:
                        break
                    try:
                        received_data = json.loads(data.decode('utf-8'))
                        image_name = received_data['image_file']

                        try:
                            path = os.path.join('img', image_name)
                            image = load_image(path)

                            prompt = f"USER: <image>\nWhat symptoms can the person recognize?\nASSISTANT:"
                            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

                            generated = model.generate(**inputs, max_new_tokens=384)
                            generated_texts = processor.batch_decode(generated, skip_special_tokens=True)

                            generated_string = list_to_string(generated_texts)
                            model_response = cut_string(generated_string)

                        except Exception as e:
                            print(f"Error processing img: {e}")
                            model_response = "Error processing img."

                        response_data = {
                            "model_response": model_response
                        }
                        conn.sendall(json.dumps(response_data).encode('utf-8'))
                    except Exception as e:
                        print(f"Error decoding JSON: {e}")
                        conn.sendall(json.dumps({"model_response": "Error decoding JSON."}).encode('utf-8'))


if __name__ == "__main__":
    listen_for_prompts()
