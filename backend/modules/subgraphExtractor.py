from modules.KnowledgeExtraction import subgraph_builder, embedding
from modules.util.kg_utils import meta_relations_dict
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from modules import llm
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
    # Use the correct knowledge graph files that exist
    kg_path = os.path.join('util', 'knowledgegraph_embeddings', 'final_kg_with_embeddings_and_index_and_name.pkl')
    emb_path = os.path.join('util', 'knowledgegraph_embeddings', 'final_kg_embeddings_tensor.pt')
    
    subgraph = subgraph_builder.SubgraphBuilder(
        kg_name_or_path=kg_path,
        kg_embeddings_path=emb_path,
        meta_relation_types_dict=meta_relations_dict,
        embedding_method=embedding.create_embedding
    )
    
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
    
    #Combine the image caption and message to the input if imageCaptioning exists:
    input_text = message
    if imgCaptioning:
        input_text = message + " " + imgCaptioning
      
    subgraph = extract_knowledge(patient_id, input_text)
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
      

      
    input, res = llm_instance.chat_with_robotdoc(patient_id, patient_info, message, node_strings, image_captioning=imgCaptioning)
    draw_graph(graph, patient_id)
    return res
      
    #Extract the content from the graph
  except Exception as e:
        input, res = llm_instance.chat_with_robotdoc(patient_id, patient_info, message, nodes_from_subgraph=None, image_captioning=imgCaptioning)
        return res
  
def processWithoutKG(patient_id, patient_info, message, imgCaptioning = None):
    input, res = llm_instance.chat_with_robotdoc(patient_id, patient_info, message, nodes_from_subgraph=None, image_captioning=imgCaptioning)
    return res
      
    
def build_structured_context(graph):
    """
    Build structured context from a NetworkX graph for LLM processing.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        list: Structured context strings for LLM
    """
    context_strings = []
    
    if not graph or len(graph.nodes()) == 0:
        return context_strings
    
    # Extract node information
    for node, data in graph.nodes(data=True):
        name = data.get('name', str(node))
        node_type = data.get('type', 'unknown')
        
        # Create structured context string
        context_string = f"Entity: {name} (Type: {node_type})"
        
        # Add additional information if available
        if 'raw_data' in data:
            raw_data = data['raw_data']
            if isinstance(raw_data, str) and len(raw_data) > 0:
                # Truncate long descriptions
                if len(raw_data) > 200:
                    raw_data = raw_data[:200] + "..."
                context_string += f" - {raw_data}"
        
        context_strings.append(context_string)
    
    # Extract edge information (relationships)
    for source, target, data in graph.edges(data=True):
        relation = data.get('relation', 'related_to')
        source_name = graph.nodes[source].get('name', str(source))
        target_name = graph.nodes[target].get('name', str(target))
        
        relationship_string = f"Relationship: {source_name} --[{relation}]--> {target_name}"
        context_strings.append(relationship_string)
    
    return context_strings
      