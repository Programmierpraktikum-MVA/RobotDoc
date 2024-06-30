from modules.KnowledgeExtraction import knowledge_extractor, subgraph_builder, trie_structure, embedding
from util.kg_utils import node_types, metadata, meta_relations_dict
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from modules.newmodel import llm

from modules.KnowledgeExtraction.trie_structure import Trie
from modules.KnowledgeExtraction.knowledge_extractor import KnowledgeExtractor
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline



fine_tuned_model =  AutoModelForTokenClassification.from_pretrained("mdecot/RobotDocNLP")
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
sympPipeline = pipeline("ner", model=fine_tuned_model, tokenizer=tokenizer,aggregation_strategy="simple")


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
def extract_knowledge(input):
    subgraph = subgraph_builder.SubgraphBuilder('util/datasets/prime_kg_nx_63960.pickle', 'util/datasets/prime_kg_embeddings_tensor_63960.pt', meta_relations_dict, embedding.create_embedding, None, None)
    edges, edge_indices = subgraph.extract_knowledge_from_kg(input, entities_list= subgraph.medNER(input))
   
    subgraph.expand_graph_with_knowledge(edge_indices)
    subgraph.save_graph('util/datasets','graph','01')
    return None


# Load the graph object
def load_graph(file_path):
  with open(file_path, 'rb') as file:
    graph = pickle.load(file)
  return graph

# Extract the content from the graph
def draw_graph(graph):
  pos = nx.spring_layout(graph)
  # Zeichnen der Knoten mit Namen
  node_labels = nx.get_node_attributes(graph, 'name')
  nx.draw(graph, pos, with_labels=True, labels=node_labels, node_color='lightblue', edge_color='pink', node_size=500, font_size=10)

  # Zeichnen der Kanten mit Beziehungen
  edge_labels = nx.get_edge_attributes(graph, 'relation')
  nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

  plt.show()

def processMessage(message):
    # Extract the knowledge from the input and create subgraph
    try:
      
      subgraph = extract_knowledge(message)
       # Load the graph object
      graph = load_graph('util/datasets/graph_01.p')
      node_strings = []
      for node in graph.nodes(data=True):
        name = node[1]['name']
        type = node[1]['type']
        context_prompt = f"The Content: {name}, {type}"
        node_strings.append(context_prompt)
      
      print(node_strings)
        
      input, res = llm.chat_with_robodoc(message, node_strings)

      return res
      
    #Extract the content from the graph
    except Exception as e:
        input, res = llm.chat_with_robodoc(message, None)
        return (f"No knowledge extraction possible with given input. \n{res}")
      
    
   
