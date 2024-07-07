import numpy as np
import faiss
import torch

from typing import Callable, Tuple, List, Optional
from gensim.parsing.preprocessing import STOPWORDS
from nltk import RegexpTokenizer
from modules.KnowledgeExtraction.trie_structure import Trie


class KnowledgeExtractor:
    def __init__(self, trie: Trie, text: str, embedding_method: Callable, init_data_method: Callable = None, entities=None):
        """
        Args:
            trie: The trie from which knowledge is extracted
            text: The context node
            embedding_method: a domain specific embedding function for the embedding of the context node
            init_data_method: an optional data initialization function
            entities: The extracted entities from the input text. If None, every word of the input text will be interpreted as an entity (excluding stopwords)
        """

        self.trie = trie
        self.text = text
        self.data = {}
        self.stopwords = STOPWORDS.union(set(['I', 'you', 'he', 'she', 'it', 'we', 'they']))
        self.context_node_embedding = None
        self.entities = None
        self.knowledge_triplets_list = None
        self.knowledge_nodes_embedding = None
        self.knowledge_indices = None
        self.context_node_embedding = self.check_emb(embedding_method(text))
        self.set_entities(entities)

        if init_data_method is not None:
            init_data_method(self.entities, self.data)

    def get_data(self, key):
        return self.data[key]

    def get_stopwords(self):
        return self.stopwords

    def get_text_embedding(self):
        return self.context_node_embedding

    def set_entities(self, entities=None):
        if entities is None:
            tokenizer = RegexpTokenizer(r"\w+")
            text_tokenized = tokenizer.tokenize(self.text)
            basic_entities = [word for word in text_tokenized if not word.lower() in self.stopwords]
            self.entities = basic_entities
        else:
            self.entities = entities

    def get_entities(self):
        return self.entities

    def check_emb(self, emb):
        if isinstance(emb, list):
            
            emb = [tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor for tensor in emb]
        

        emb = np.array(emb)
        # if type(emb) != np.ndarray:
        #     emb = np.array(emb)
        if emb.ndim == 1:
            emb = emb[None, :]
        return emb

    def search_neighborhood(self, num_neighbors: int) -> Tuple[Optional[np.chararray], Optional[np.ndarray]]:
        """

        Args:
            num_neighbors: number of nodes to be includes in the neighborhood

        Returns:
            neighborhood: A list of edges of the newly extracted neighborhood
            neighborhood_indices: The corresponding indices to the neighborhood's edges
        """
        if self.knowledge_triplets_list is None:
            return None, None

        # extract the doublet triplets
        triplets = np.char.array([self.knowledge_triplets_list[:, 0], self.knowledge_triplets_list[:, 1]]).T
        cleared_triplets, triplets_indicies = np.unique(triplets, return_index=True, axis=0)
        self.knowledge_triplets_list = self.knowledge_triplets_list[triplets_indicies]
        self.knowledge_nodes_embedding = self.knowledge_nodes_embedding[triplets_indicies]

        neighborhood_indices = None

        # reduce k
        if len(self.knowledge_triplets_list) < num_neighbors:
            # extract best triplets
            neighborhood = self.knowledge_triplets_list
            if self.knowledge_indices is not None:
                self.knowledge_indices = self.knowledge_indices[triplets_indicies]
                neighborhood_indices = self.knowledge_indices

        else:
            # search for best triplets
            # cpu search
            d = self.context_node_embedding.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(np.reshape(self.knowledge_nodes_embedding, (self.knowledge_nodes_embedding.shape[0], -1)))
            D, I = index.search(self.context_node_embedding, num_neighbors)

            # extract best triplets
            neighborhood = self.knowledge_triplets_list[I[0]]

            # delete extracted triplets
            mask = np.ones(len(self.knowledge_triplets_list), dtype=bool)
            mask[I[0]] = False
            self.knowledge_triplets_list = self.knowledge_triplets_list[mask]
            self.knowledge_nodes_embedding = self.knowledge_nodes_embedding[mask]

            if self.knowledge_indices is not None:
                self.knowledge_indices = self.knowledge_indices[triplets_indicies]
                neighborhood_indices = self.knowledge_indices[I[0]]
                self.knowledge_indices = self.knowledge_indices[mask]

        return neighborhood, neighborhood_indices

    def new_hop(self, neighborhood, extracted_edges, extracted_edge_indices=None, neighbors_per_hop=10) -> Tuple[Optional[np.chararray], Optional[np.chararray], Optional[np.ndarray]]:
        """
        Args:
            neighborhood: The most recently extracted trie neighborhood's edges
            extracted_edges: Top-rated edges (based on word embeddings similarity), will creat the subgraph later on
            extracted_edge_indices: The corresponding graph indices to extracted_edges (optional)
            neighbors_per_hop: Number of edges to be extracted each hop

        Returns:
            neighborhood: The newly extracted neighborhood
            extracted_edges: The updated top-rated edges
            extracted_edge_indices: The corresponding graph indices to extracted_edges (optional)
        """

        if neighborhood is not None:
            new_knowledge_edges, new_knowledge_embeddings, new_knowledge_indices = self.pull_from_kg_trie(entities=neighborhood[:, 1])
            self.entities = np.concatenate([self.entities, neighborhood[:, 1]])
        else:
            new_knowledge_edges, new_knowledge_embeddings, new_knowledge_indices = self.pull_from_kg_trie()


        #print(f"Here is return of new_hop: {type(new_knowledge_embeddings[0])}")

        self.update_knowledge(new_knowledge_edges, new_knowledge_embeddings, new_knowledge_indices)

        neighborhood, neighborhood_indices = self.search_neighborhood(neighbors_per_hop)

        if extracted_edges is None:
            extracted_edges = neighborhood
        else:
            extracted_edges = np.concatenate((extracted_edges, neighborhood), axis=0)

        if neighborhood_indices is not None:
            if extracted_edge_indices is None:
                extracted_edge_indices = neighborhood_indices
            else:
                extracted_edge_indices = np.concatenate((extracted_edge_indices, neighborhood_indices), axis=0)

        return neighborhood, extracted_edges, extracted_edge_indices

    def pull_from_kg_trie(self, entities=None) -> Tuple[List[str], List[torch.Tensor], List[int]]:
        """
        Args:
            entities: A list of word entities for querying the trie

        Returns:
            new_knowledge_edges: The extracted edges from the trie
            new_knowledge_embeddings: The corresponding node embeddings to new_knowledge_edges
            new_knowledge_indices THe corresponding indices to new_knowledge_edges
        """
        if entities is None:
            entities = self.entities

        if type(entities) == type(""):
            entities = [entities]

        new_knowledge_indices = []
        new_knowledge_edges = []
        new_knowledge_embeddings = []

        for entity in entities:
            #print(entity)
            triplet, embedding, triplet_indices = self.trie.query(entity, avoid_cycles=True)
            new_knowledge_edges += triplet
            new_knowledge_embeddings += embedding
            new_knowledge_indices += triplet_indices

        #print(f"Here is return of pull_from_kg_trie: {type(new_knowledge_embeddings[0])}")
        return new_knowledge_edges, new_knowledge_embeddings, new_knowledge_indices

    def extract_subgraph_from_query(self, hops=4, neighbors_per_hop=100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Args:
            hops:  Number of pulls from the trie
            neighbors_per_hop: How many nodes should be extracted each hop

        Returns:
            extracted_edges: Top-rated extracted KG edges
            extracted_edge_indices: The corresponding indices to extracted_edges
        """

        neighborhood, extracted_edges, extracted_edge_indices = None, None, None

        for hop in range(hops):
            neighborhood, extracted_edges, extracted_edge_indices = self.new_hop(neighborhood, extracted_edges,
                                                                                 extracted_edge_indices, neighbors_per_hop=neighbors_per_hop)

            if extracted_edges is None:
                break

        return extracted_edges, extracted_edge_indices

    def update_knowledge(self, new_knowledge_edges: list, new_embeddings, new_indices=None):
        """This method accumulates the extracted knowledge by adding the newly pulled tire knowledge to
            self.knowledge_triplets_list, self.knowledge_nodes_embedding and self.knowledge_indices"""
        #print(type(new_embeddings[0]))

        if len(new_knowledge_edges) > 0:
            if self.knowledge_triplets_list is None:
                self.knowledge_triplets_list = np.char.array(new_knowledge_edges)
            else:
                self.knowledge_triplets_list = np.concatenate((self.knowledge_triplets_list, new_knowledge_edges), axis=0)

            if self.knowledge_nodes_embedding is None:
                self.knowledge_nodes_embedding = self.check_emb(new_embeddings)
            else:
                self.knowledge_nodes_embedding = np.concatenate(
                    (self.knowledge_nodes_embedding, self.check_emb(new_embeddings)), axis=0)

            if new_indices is not None:
                if self.knowledge_indices is None:
                    self.knowledge_indices = np.asarray(new_indices)
                else:
                    self.knowledge_indices = np.concatenate(
                        (self.knowledge_indices, np.asarray(new_indices)), axis=0)
