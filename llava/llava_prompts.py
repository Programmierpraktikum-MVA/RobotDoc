def category_prompt():
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