import torch
import openai
import os
openai.api_key = os.environ.get("OPENAI_API_KEY")

def create_embedding(text):
    #Erstelle Sentence embeddings mit openaiApi, analog zu Node embeddings im KG
    embedding = torch.tensor(openai.Embedding.create(input=[text], model="text-embedding-ada-002")['data'][0]['embedding'], dtype=torch.float32)
    # Normiere auf die L2-Norm edit: ne doch lieber nicht
    #embedding = embedding / embedding.norm(p=2)
    
    #tensor dimension = 1 ; tensor size = 1536
    
    
    return embedding