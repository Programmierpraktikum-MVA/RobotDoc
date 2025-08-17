import os
import json
import torch
import socket
from transformers import LlavaNextProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
from PIL import Image


# ---------------------------
# CONFIG
# ---------------------------
MODE = 1  # 1 = simple mode, 2 = prompt-tree mode

model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
save_directory = "./model"  # Specify your desired save directory

# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ---------------------------
# HELPERS
# ---------------------------
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
    cutpoint = "ASSISTANT:"
    start_idx = llava_out.find(cutpoint)
    if start_idx != -1:
        return llava_out[start_idx + len(cutpoint):]
    return llava_out

def give_category():
    """Stage 1: Classification"""
    return (
        "USER: <image>\n"
        "You are a medical image classification expert.\n"
        "Classify the image into exactly one of the following categories:\n"
        "- Skin: Lesions, texture, color patterns, ABCDE, melanoma indicators\n"
        "- Radiology: 2D imaging of internal body structures (e.g., X-ray, CT, MRI)\n"
        "- Wound: Tissue condition, infection, moisture, wound edges, healing phase\n"
        "\n"
        "If the image clearly does not belong to any of these, consider whether it might still be Radiology. If not, respond with: Other\n"
        "Respond with only the category name.\n"
        "ASSISTANT:"
    )

def get_prompt(category):
    """Stage 2: Category-specific prompt"""
    prompts = {
        "Skin": (
            "USER: <image>\n"
            "Analyze the visible skin lesion using standard dermatological criteria. Describe visible features:\n"
            "- Efflorescence: Type and morphology (e.g., papule, nodule, ulcer)\n"
            "- Color: Dominant hues + estimated RGB values\n"
            "- Symmetry: Symmetric or asymmetric shape\n"
            "- Border: Sharp vs. blurred edges and regularity\n"
            "- Size: Approximate dimensions in mm/cm\n"
            "- Location: Anatomical site (e.g., face, back, arm)\n"
            "- Pattern: Distribution (isolated, grouped, linear, etc.)\n"
            "Do not make a diagnosis or recommendation.\n"
            "ASSISTANT:"
        ),
        "Radiology": (
            "USER: <image>\n"
            "Provide a detailed and objective visual description of the radiological scan. For each visible structure or area, describe:\n"
            "- Modality (e.g., X-ray, CT, MRI)\n"
            "- Body region imaged\n"
            "- All observed findings: size, shape, density, location, opacity, and any abnormal or unusual features\n"
            "- Specifically note any lung opacities, consolidations, ground-glass patterns, or air bronchograms\n"
            "- Presence of artifacts or imaging issues\n"
            "If no abnormalities are seen, explicitly state “No abnormal findings detected.”\n"
            "Avoid vague phrases such as “no problem.”\n"
            "Do not provide clinical diagnosis or treatment recommendations. Focus solely on describing the findings objectively.\n\n"
            "Example 1 (Normal Chest X-ray):\n"
            "Modality: Chest X-ray\n"
            "Region: Thorax\n"
            "Findings: Clear lung fields bilaterally with normal vascular markings. Cardiac silhouette normal in size and shape. No pleural effusion or pneumothorax. Bone structures intact. No artifacts.\n\n"
            "Example 2 (Chest X-ray with Pneumonia):\n"
            "Modality: Chest X-ray\n"
            "Region: Thorax\n"
            "Findings: Patchy, ill-defined opacity in the right lower lung zone measuring approximately 5 cm. Consolidation with air bronchograms visible. No pleural effusion. Cardiac silhouette normal. No imaging artifacts.\n\n"
            "Example 3 (CT Chest with Ground-glass Opacity):\n"
            "Modality: CT Chest\n"
            "Region: Thorax\n"
            "Findings: Multiple areas of ground-glass opacity bilaterally, predominantly in the lower lobes. No discrete nodules or masses. Mild thickening of bronchial walls. No pleural effusion or pneumothorax. No artifacts.\n\n"
            "Example 4 (Chest X-ray with Atelectasis):\n"
            "Modality: Chest X-ray\n"
            "Region: Thorax\n"
            "Findings: Linear opacity with volume loss seen in the left lower lung zone consistent with atelectasis. No consolidations or pleural effusions. No artifacts.\n"
            "ASSISTANT:"
        ),
        "Wound": (
            "USER: <image>\n"
            "Visually assess the wound using the TIME framework. Describe what is visible:\n"
            "- Tissue: Type (e.g., necrotic, granulation), color, texture\n"
            "- Infection/Inflammation: Redness, exudate, swelling\n"
            "- Moisture: Dry, moist, wet, presence of slough or exudate\n"
            "- Edges: Defined/undefined, undermining, epithelialization\n"
            "If visible, estimate wound size and depth. Do not recommend treatment.\n"
            "ASSISTANT:"
        ),
        "Other": (
            "USER: <image>\n"
            "Describe the image using objective visual terms. Include:\n"
            "- Visible structures or objects\n"
            "- Color, shape, and texture patterns\n"
            "- Any abnormalities or unusual features\n"
            "Avoid assumptions or medical interpretations.\n"
            "ASSISTANT:"
        )
    }

    return prompts.get(category, prompts["Other"])

def listen_for_prompts():
    print("Loading model into VRAM...")
    try:
        processor = LlavaNextProcessor.from_pretrained(model_name)
        #We distribute the model to only one GPU, as there is only one GPU available.
        model = LlavaForConditionalGeneration.from_pretrained(model_name, quantization_config=quantization_config, low_cpu_mem_usage=True, torch_dtype=torch.float16).to("cuda")
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
                            
                            if MODE == 1:
                                prompt = f"USER: <image>\nYou are a medical image analyst. What symptoms and abnormalities do you recognize in the given image?\nASSISTANT:"
                                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

                                generated = model.generate(**inputs, max_new_tokens=384)
                                generated_texts = processor.batch_decode(generated, skip_special_tokens=True)

                                generated_string = list_to_string(generated_texts)
                                model_response = cut_string(generated_string)

                            elif MODE == 2:
                                prompt = give_category()
                                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                                generated_classification = model.generate(**inputs, max_new_tokens=20)
                                classification_text = processor.batch_decode(generated_classification, skip_special_tokens=True)
                                classification_string = list_to_string(classification_text)
                                category = cut_string(classification_string)

                                prompt = get_prompt(category)
                                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                                temp = 0.4 if category == "Radiology" else 0.3
                                generated = model.generate(**inputs, max_new_tokens=280, temperature=temp, top_p=0.9, do_sample=True)
                                generated_texts = processor.batch_decode(generated, skip_special_tokens=True)

                                generated_string = list_to_string(generated_texts)
                                model_response = cut_string(generated_string)
                            else:
                                print(f"Error no assigned mode detected with Mode: {MODE}")
                                model_response = "Error processing img."

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
