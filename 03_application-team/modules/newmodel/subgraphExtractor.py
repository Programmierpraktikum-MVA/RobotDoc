from modules.KnowledgeExtraction import knowledge_extractor, subgraph_builder, trie_structure, embedding
from util.kg_utils import node_types, metadata, meta_relations_dict
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from modules.newmodel import llm
import os

from modules.KnowledgeExtraction.trie_structure import Trie
from modules.KnowledgeExtraction.knowledge_extractor import KnowledgeExtractor
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline



fine_tuned_model =  AutoModelForTokenClassification.from_pretrained("mdecot/RobotDocNLP")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
sympPipeline = pipeline("ner", model=fine_tuned_model, tokenizer=tokenizer,aggregation_strategy="simple")

llm_instance = llm.LLM()

def symptomNER(text):
        prev = ''
        symptoms = []
        output = sympPipeline(text)
        prev_ent = ''
        for g in output:
            entity =  g['entity_group']
            w = g['word']
            
            if(entity=='Sign_symptom'):
                if(prev_ent=='Sign_symptom'):
                    if(w.startswith('##')):
                        prev = symptoms.pop()
                        new=prev+w.replace('#','')
                        symptoms.append(new)
                    else:
                        symptoms.append(w)
                else:
                    symptoms.append(w)
            prev_ent=entity
        return symptoms

# Extract the knowledge from the input and create subgraph
def extract_knowledge(patient_id, input):
    subgraph = subgraph_builder.SubgraphBuilder(os.path.join('util', 'datasets', 'prime_kg_nx_63960.pickle'), os.path.join('util', 'datasets', 'prime_kg_embeddings_tensor_63960.pt'), meta_relations_dict, embedding.create_embedding, None, None)
    graph_filename = os.path.join('util', 'datasets', f'graph_{patient_id}.p')
    if os.path.exists(graph_filename):
      with open(graph_filename, 'rb') as f:
        subgraph.nx_subgraph = pickle.load(f)
    
    edges, edge_indices = subgraph.extract_knowledge_from_kg(input, entities_list = subgraph.medNER(input))
    subgraph.expand_graph_with_knowledge(edge_indices)

    subgraph.save_graph(os.path.join('util', 'datasets'),'graph', patient_id)
    
    return None



# Load the graph object
def load_graph(file_path):
  with open(file_path, 'rb') as file:
    graph = pickle.load(file)
  return graph

# plt.switch_backend('Agg')
plt.switch_backend('Agg')

def draw_graph(graph, patient_id):
    pos = nx.spring_layout(graph)
    # Draw the nodes with names
    node_labels = nx.get_node_attributes(graph, 'name')
    nx.draw(graph, pos, with_labels=True, labels=node_labels, node_color='lightblue', edge_color='pink', node_size=500, font_size=10)

    # Draw the edges with relations


    # Save the graph as an image
    plt.savefig(os.path.join('static', 'img', f'graph_{patient_id}.png'))

    plt.close()  # Close the figure to free up resources


def processMessage(patient_id, patient_info, message, imgCaptioning = None):
  
    # Extract the knowledge from the input and create subgraph
  try:

      
    subgraph = extract_knowledge(patient_id, message)
     # Load the graph object
    graph_filename = os.path.join('util', 'datasets', f'graph_{patient_id}.p')

    graph = load_graph(graph_filename)
    node_strings = []
    for node in graph.nodes(data=True):
      name = node[1]['name']
      type = node[1]['type']
      context_prompt = f"The Content: {name}, {type}"
      node_strings.append(context_prompt)
      print(name)
      

      
    input, res = llm_instance.chat_with_robodoc(patient_id, patient_info, message, node_strings, image_captioning=imgCaptioning)
    draw_graph(graph, patient_id)
    return res
      
    #Extract the content from the graph
  except Exception as e:
        input, res = llm_instance.chat_with_robodoc(patient_id, patient_info, message, nodes_from_subgraph=None, image_captioning=imgCaptioning)
        return res
  
def processWithoutKG(patient_id, patient_info, message, imgCaptioning = None):
    input, res = llm_instance.chat_with_robodoc(patient_id, patient_info, message, nodes_from_subgraph=None, image_captioning=imgCaptioning)
    return res
      
    
