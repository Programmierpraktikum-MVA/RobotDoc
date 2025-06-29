import os
import pickle
import torch

import networkx as nx
import torch_geometric.transforms as T

from typing import List, Tuple, Callable, Union
from datasets import load_dataset
from numpy import ndarray
from torch_geometric.data import HeteroData

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


from modules.KnowledgeExtraction.trie_structure import Trie
from modules.KnowledgeExtraction.knowledge_extractor import KnowledgeExtractor


class SubgraphBuilder:
    def __init__(self,
                 kg_name_or_path: str,
                 kg_embeddings_path: str,
                 #dataset_name_or_path: str,
                 meta_relation_types_dict: dict,
                 embedding_method: Callable,
                 trie_path: str = None,
                 trie_save_path: str = None,
                 ):
        """

        Args:
            kg_name_or_path: The path (or name) to the KG
            kg_embeddings_path: The path to the KG's nodes' embeddings
            dataset_name_or_path: Path (or name) to the dataset being used
            meta_relation_types_dict: A dict of the form: 'relation_type': ('source_type', 'relation_type', target_type') containing all edge types needed for creating self.hetero_data
            embedding_method: A callable. Domain specific embedding method to compute sentence embeddings. Should be the same embedding model that was used to compute self.node_embeddings
            trie_path: Optional. PAth to a precomputed trie
            trie_save_path: Optional. A target path to save newly computed trie
        """

        # load KG, node embeddings a trie and a dataset
        self.kg = self.load_kg(kg_name_or_path)

        self.node_embeddings = torch.load(kg_embeddings_path)

        self.trie = self.set_trie(kg_name_or_path, trie_path, trie_save_path)

        #self.dataset = load_dataset(dataset_name_or_path)

        self.nx_subgraph = nx.Graph()

        self.hetero_data = HeteroData()

        self.meta_relation_types_dict = meta_relation_types_dict

        self.embedding_method = embedding_method

        #load & init model & pipeline
        
        self.fine_tuned_model =  AutoModelForTokenClassification.from_pretrained("mdecot/RobotDocNLP")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        self.sympner = pipeline("ner", model=self.fine_tuned_model, tokenizer=self.tokenizer,aggregation_strategy="simple")

        self.ner = pipeline("token-classification", model="medical-ner-proj/bert-medical-ner-proj")


    def set_trie(self, kg_name_or_path, trie_path, trie_save_path=None):
        if trie_path is not None:
            return pickle.load(open(trie_path, 'rb'))

        if "conceptnet" in kg_name_or_path:
            trie = self.build_trie_from_conceptnet(trie_save_path)
        else:
            trie = self.build_trie_from_nx_kg(trie_save_path)
        return trie

    def load_kg(self, kg_name_or_path):
        if "conceptnet" in kg_name_or_path:
            kg = load_dataset('')

        else:
            kg = pickle.load(open(kg_name_or_path, 'rb'))

        return kg
        
    def medNER(self, text):
        prev = ''
        entities = []
        output = self.ner(text)
        for g in output:
            #if g['entity_group'] == 'Sign_symptom':
            w = g['word']
        
            if w.startswith('##'):
                prev = entities.pop()
                new = prev +w.replace('#', '')
                entities.append(new)
            else:
                entities.append(w)
        return entities
    
    def symptomNER(self, text):
        prev = ''
        symptoms = []
        output = self.sympner(text)
        prev_ent = ''
        for g in output:
            entity =  g['entity_group']
            w = g['word']
            
            if(entity=='Sign_symptom'):
                if(prev_ent=='Sign_symptom'):
                    if(w.startswith('##')):
                        prev = symptoms.pop()
                        new=prev+w.replace('#','')
                    else:
                        symptoms.append(w)
                else:
                    symptoms.append(w)
            prev_ent=entity
        return symptoms
    


    def build_trie_from_nx_kg(self, save_path=None):
        """
        If you wish to use ConceptNet (or a similar KG dataset istead of a nx graph please use the function
        insert_dataset(dataset, embeddings) from the trie_structure file instead of insert_knowledge_graph(...)
        """
        
        trie = Trie()
        trie.insert_knowledge_graph(self.kg, 'name', store_node_index=True)

        if save_path is not None:
            trie.save_trie(save_path)

        return trie

    def build_trie_from_conceptnet(self, save_path=None):
        trie = Trie()
        trie.insert_dataset(self.kg, self.node_embeddings)

        if save_path is not None:
            trie.save_trie(save_path)

        return trie

    def extract_knowledge_from_kg(self, query_str: str, hops: int = 2, neighbors_per_hop: int = 10, entities_list: List[str] = None) -> Union[Tuple[None, None], Tuple[Union[ndarray, ndarray], Union[ndarray, ndarray, None]]]:
        """
        Args:
            neighbors_per_hop:
            hops:
            query_str: The string which we want to enrich with the subgraph
            entities_list: Optional. A preprocessed list of entities extracted from the query_str. If None, every word of
            query_str (excluding stopwords) will be processed.

        Returns:
            extracted_edges: A list of triplets of the extracted edges from the trie in the form [source_node, target_node, relation_type]
            extracted_edge_indices: Optional. The corresponding indices to the edges in extracted_edges
        """

        # Find relevant subgraphs using DiaTransNet's trie_structure
        knowledge_extractor = KnowledgeExtractor(self.trie, query_str, embedding_method=self.embedding_method, entities=entities_list)
        

        if len(knowledge_extractor.entities) == 0:
            return None, None

        extracted_edges, extracted_edge_indices = knowledge_extractor.extract_subgraph_from_query(hops=hops, neighbors_per_hop=neighbors_per_hop)

        if extracted_edges is None:
            return None, None

        return extracted_edges, extracted_edge_indices

    def convert_nx_to_hetero_data(self, edge_uid_offset=0) -> Tuple[HeteroData, int]:
        """

        Args:
            edge_uid_offset: Optional. Used for building the edge_uid attribute per edge type in the HeteroData, which keeps track of all unique edges in the graph.
                When this method is called within a loop over many subgraphs and a universal edge indexing is beneficiary please use as follows:
                hetero_data, edge_uid_offset = convert_nx_to_hetero_data(edge_uid_offset).

        Returns:
            data: the HeteroData object created from the input graph
            edge_uid_offset: the updated edge_uid_offset


        """
        node_types_embeddings_dict = {}
        node_types_uids_dict = {}
        edge_types_index_dict = {}
        edge_types_uids_dict = {}

        # Iterate over all edges:
        for index, (s, t, edge_attr) in enumerate(self.nx_subgraph.edges(data=True)):

            relation = self.meta_relation_types_dict[edge_attr['relation']]

            # Source node
            s_node = self.nx_subgraph.nodes[s]
            s_node_type = s_node['type']
            s_node_embedding = s_node['embedding']
            s_uid = s_node['index']
            if s_node_embedding.dim() == 2:
                s_node_embedding = torch.squeeze(s_node_embedding, dim=1)

            # Target node
            t_node = self.nx_subgraph.nodes[t]
            t_node_type = t_node['type']
            t_node_embedding = t_node['embedding']
            t_uid = t_node['index']
            if t_node_embedding.dim() == 2:
                t_node_embedding = torch.squeeze(t_node_embedding, dim=1)

            # The graph is undirected - make sure edges are uniform from calling the ToUndirected() transformation later
            if s_node_type != relation[0]:
                s_node_type, t_node_type = t_node_type, s_node_type
                s_node_embedding, t_node_embedding = t_node_embedding, s_node_embedding
                s_uid, t_uid = t_uid, s_uid

            # Add source node to the graph's nodes and record it's index
            if s_node_type not in node_types_embeddings_dict:
                node_types_embeddings_dict[s_node_type] = []
                node_types_uids_dict[s_node_type] = []
                s_node_index = len(node_types_embeddings_dict[s_node_type])
                node_types_embeddings_dict[s_node_type].append(s_node_embedding)
                node_types_uids_dict[s_node_type].append(s_uid)

            elif s_uid not in node_types_uids_dict[s_node_type]:
                s_node_index = len(node_types_embeddings_dict[s_node_type])
                node_types_embeddings_dict[s_node_type].append(s_node_embedding)
                node_types_uids_dict[s_node_type].append(s_uid)

            else:
                s_node_index = node_types_uids_dict[s_node_type].index(s_uid)

            # Add target node to the graph's nodes and record it's index
            if t_node_type not in node_types_embeddings_dict:
                node_types_embeddings_dict[t_node_type] = []
                node_types_uids_dict[t_node_type] = []
                t_node_index = len(node_types_embeddings_dict[t_node_type])
                node_types_embeddings_dict[t_node_type].append(t_node_embedding)
                node_types_uids_dict[t_node_type].append(t_uid)

            elif t_uid not in node_types_uids_dict[t_node_type]:
                t_node_index = len(node_types_embeddings_dict[t_node_type])
                node_types_embeddings_dict[t_node_type].append(t_node_embedding)
                node_types_uids_dict[t_node_type].append(t_uid)

            else:
                t_node_index = node_types_uids_dict[t_node_type].index(t_uid)

            # Add edge to the graph's edges
            if relation not in edge_types_index_dict:
                edge_types_index_dict[relation] = []
                edge_types_index_dict[relation].append([s_node_index, t_node_index])
                edge_types_uids_dict[relation] = []
                edge_types_uids_dict[relation].append(edge_uid_offset)
                edge_uid_offset += 1

            elif [s_node_index, t_node_index] not in edge_types_index_dict[relation]:
                edge_types_index_dict[relation].append([s_node_index, t_node_index])
                edge_types_uids_dict[relation].append(edge_uid_offset)
            edge_uid_offset += 1

        # Iterate over nodes with no neighbors and add them tot eh graph:
        nodes_with_no_neighbors = [self.nx_subgraph.nodes[node] for node in self.nx_subgraph.nodes() if len(list(self.nx_subgraph.neighbors(node))) == 0]
        for node in nodes_with_no_neighbors:
            node_type = node['type']
            node_embedding = node['embedding']
            node_uid = node['index']
            if node_embedding.dim() == 2:
                node_embedding = torch.squeeze(node_embedding, dim=1)
            if node_type not in node_types_embeddings_dict:
                node_types_embeddings_dict[node_type] = []
                node_types_uids_dict[node_type] = []
                node_types_embeddings_dict[node_type].append(node_embedding)
                node_types_uids_dict[node_type].append(node_uid)

            elif node_uid not in node_types_uids_dict[node_type]:
                node_types_embeddings_dict[node_type].append(node_embedding)
                node_types_uids_dict[node_type].append(node_uid)

        # Create node feature tensor x per node type
        for n_type in node_types_embeddings_dict.keys():
            self.hetero_data[n_type].x = torch.stack(node_types_embeddings_dict[n_type], dim=0).type("torch.FloatTensor")
            self.hetero_data[n_type].node_uid = torch.tensor(node_types_uids_dict[n_type])

        # Create edge_index tensor per edge type
        for e_type in edge_types_index_dict.keys():
            self.hetero_data[e_type].edge_index = torch.transpose(torch.tensor(edge_types_index_dict[e_type]), 0, 1)
            self.hetero_data[e_type].edge_uid = torch.tensor(edge_types_uids_dict[e_type])

        self.hetero_data = T.ToUndirected(merge=False)(self.hetero_data)

        return self.hetero_data, edge_uid_offset

    def expand_graph_with_knowledge(self, extracted_edge_indices: [list]) -> nx.Graph:
        for source, target in extracted_edge_indices:
            source_node = self.kg.nodes[source]
            target_node = self.kg.nodes[target]
            relation = self.kg[source][target]['relation']

            self.nx_subgraph.add_node(source, **source_node)
            self.nx_subgraph.add_node(target, **target_node)
            self.nx_subgraph.add_edge(source, target, relation=relation)

        return self.nx_subgraph

    def save_graph(self, destination_dir, filename, index):
        pickle.dump(self.nx_subgraph, open(os.path.join(destination_dir, f'{filename}_{index}.p'), "wb"))

    def save_hetero_data(self, destination_dir, filename, index):
        pickle.dump(self.hetero_data, open(os.path.join(destination_dir, f'{filename}_{index}.p'), "wb"))