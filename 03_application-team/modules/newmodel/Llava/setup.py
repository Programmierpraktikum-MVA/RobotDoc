import os
import io
from io import BytesIO
import json
import torch
import socket
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from PIL import Image
from huggingface_hub import login

# Load configuration data
config_data = json.load(open("./config.json"))
HF_TOKEN = config_data["HF_TOKEN"]

model_name = "llava-hf/llava-1.5-7b-hf"
save_directory = "./model"  # Specify your desired save directory

# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

# Global variable to track if model is loaded
model_loaded = False
processor = None
model = None

def unload_model():
    global model, model_loaded
    if model_loaded:
        del processor
        del model
        torch.cuda.empty_cache()
        model_loaded = False
        print("Model unloaded from VRAM.")

def download_and_save_model(model_name, save_directory):
    try:
        if os.path.exists(save_directory):
            print("Model already downloaded")
            return True
        
        # Load the processor and model with adapters on top
        processor = AutoProcessor.from_pretrained(model_name)

        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            use_auth_token=HF_TOKEN
        )

        # Save model to disk
        model.save_pretrained(save_directory)

        print(f"Model saved to {save_directory}")

        unload_model()

        return True

    except Exception as e:
        print(f"Failed to download and save model: {str(e)}")
        return False

def load_model():
    global processor, model, model_loaded
    if not model_loaded:
        print("Loading model into VRAM...")
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            # Load the base
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                use_auth_token=HF_TOKEN
            )
            model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")

#transform output list into string
def list_to_string(llava_out):
    if not isinstance(llava_out, list):
        raise ValueError("Input hast to be a list.")

    return ' '.join(str(item) for item in llava_out)

#cut string for the relevant part:
def cut_string(llava_out):
  cutpoint = "ASSISTANT: "
  start_idx = llava_out.find(cutpoint)
  if start_idx != -1:
    return llava_out[start_idx + len(cutpoint):]
  else:
    return ""

def image_captioning_with_robodoc(img):
    global model_loaded, generated_texts

    try:
        load_model()

        prompt = f"USER: <image>\nWhat symptoms can the person recognize?\nASSISTANT:"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")

        generated = model.generate(**inputs, max_new_tokens=384)
        generated_texts = processor.batch_decode(generated, skip_special_tokens=True)

    except Exception as e:
        print(f"Exception occurred during model generation: {str(e)}")
        model_response = "Sorry, I couldn't process your request at the moment."

    finally:
        unload_model()

    generated_string = list_to_string(generated_texts)
    generated_string = cut_string(generated_string)

    return generated_string

def load_image_from_bytes(image_bytes):
    img_stream = BytesIO(image_bytes)
    image = Image.open(img_stream).convert("RGB")
    return image

def listen_for_prompts():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 65533))
        s.listen()
        print("Waiting for connections...")
        while True:  # Keep listening for connections indefinitely
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:  # Keep processing prompts within the connection
                    data = conn.recv(8192)
                    if not data:
                        break
                    try:
                        received_data = json.loads(data.decode('utf-8'))
                        image_hex = received_data['image_file']
                        try:
                            image_bytes = bytes.fromhex(image_hex)
                            #image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            image = load_image_from_bytes(image_bytes)
                            model_response = image_captioning_with_robodoc(image)
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
    # Download and save the model if not already saved
    if not download_and_save_model(model_name, save_directory):
        print("Failed to download and save model. Exiting...")
    else:
        # Start listening for prompts
        listen_for_prompts()
