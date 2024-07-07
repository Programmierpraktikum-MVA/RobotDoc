# Base Modell laden und testen

import io
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
from PIL import Image
from huggingface_hub import login
import os

#mein privater hf token (bitte den nicht pushen)
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)
# Load the base model with adapters on top
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    use_auth_token=hf_token
)

"""Lade Testbild"""

def load_image(image_file):
  image = Image.open(image_file).convert("RGB")
  return image


#hier kann dann alternativ das Bild aus der Datenbank geladen werden
image = load_image("bildname.jpeg") #png, jpg und jpeg sollten alle klappen

"""transform output list into string"""
def list_to_string(llava_out):
    if not isinstance(llava_out, list):
        raise ValueError("Input muss eine liste sein.")

    return ' '.join(str(item) for item in llava_out)

"""cut string for the relevant part:"""
def cut_string(llava_out):
  cutpoint = "ASSISTANT: "
  start_idx = llava_out.find(cutpoint)
  if start_idx != -1:
    return llava_out[start_idx + len(cutpoint):]
  else:
    return ""


# """prepare image and prompt for the model
# """

# prompt = f"USER: <image>\nWhat symptoms can the person recognize?\nASSISTANT:"
# inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

# # Generate token IDs
# generated = model.generate(**inputs, max_new_tokens=384)

# # Decode back into text
# generated_texts = processor.batch_decode(generated, skip_special_tokens=True)
# generated_string = list_to_string(generated_texts)



# "output, der an Llama3 weitergegeben werden soll:"

# generated_string = cut_string(generated_string)

# print(generated_string)
def load_image_from_bytes(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

def inference(img):
   prompt = f"USER: <image>\nWhat symptoms can the person recognize?\nASSISTANT:"
   inputs = processor(text=prompt, images=img, return_tensors="pt").to("cuda")

   generated = model.generate(**inputs, max_new_tokens=384)

   generated_texts = processor.batch_decode(generated, skip_special_tokens=True)
   del processor
   del model
   torch.cuda.empty_cache()
   generated_string = list_to_string(generated_texts)
   generated_string = cut_string(generated_string)
   return generated_string


   