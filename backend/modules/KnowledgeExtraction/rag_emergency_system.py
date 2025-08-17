import pickle
import torch
import networkx as nx
from typing import Dict, Any, List, Optional
from modules.KnowledgeExtraction import subgraph_builder, embedding
from modules.MedicalAnalysis.emergency_detection import analyze_medical_urgency
from modules.util.kg_utils import meta_relations_dict


class RAGEmergencySystem:
    """
    RAG (Retrieval-Augmented Generation) system combined with emergency detection
    for medical knowledge extraction and emergency response.
    """
    
    def __init__(self, kg_path: str, emb_path: str):
        """
        Initialize the RAG Emergency System.
        
        Args:
            kg_path: Path to the knowledge graph pickle file
            emb_path: Path to the embeddings tensor file
        """
        self.kg_path = kg_path
        self.emb_path = emb_path
        
        # Load knowledge graph
        with open(kg_path, 'rb') as f:
            self.kg = pickle.load(f)
        
        # Load embeddings
        self.embeddings = torch.load(emb_path)
        
        # Initialize subgraph builder
        self.subgraph_builder = subgraph_builder.SubgraphBuilder(
            kg_name_or_path=kg_path,
            kg_embeddings_path=emb_path,
            meta_relation_types_dict=meta_relations_dict,
            embedding_method=embedding.create_embedding
        )
        
        # Create index mappings
        self._create_index_mappings()
    
    def _create_index_mappings(self):
        """Create mappings between node indices and node IDs."""
        self.index_to_node_id = {}
        self.node_id_to_index = {}
        
        for node_id in self.kg.nodes():
            node_data = self.kg.nodes[node_id]
            if 'index' in node_data:
                index = node_data['index']
                self.index_to_node_id[index] = node_id
                self.node_id_to_index[node_id] = index
    
    def process_query(self, query: str, patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a medical query with RAG and emergency detection.
        
        Args:
            query: The medical query to process
            patient_info: Optional patient information
            
        Returns:
            Dictionary containing emergency analysis and RAG results
        """
        # First, check for emergency
        emergency_analysis = analyze_medical_urgency(query)
        
        # Extract knowledge using RAG
        entities = self.subgraph_builder.medNER(query)
        edges, edge_indices = self.subgraph_builder.extract_knowledge_from_kg(query, entities_list=entities)
        
        # Process subgraph if edges found
        subgraph_info = None
        if edge_indices is not None:
            valid_edges = []
            for source_idx, target_idx in edge_indices:
                if source_idx in self.index_to_node_id and target_idx in self.index_to_node_id:
                    source_node_id = self.index_to_node_id[source_idx]
                    target_node_id = self.index_to_node_id[target_idx]
                    valid_edges.append((source_node_id, target_node_id))
            
            if valid_edges:
                # Reset subgraph
                self.subgraph_builder.nx_subgraph = nx.Graph()
                self.subgraph_builder.expand_graph_with_knowledge(valid_edges)
                subgraph = self.subgraph_builder.nx_subgraph
                
                # Filter medical entities
                medical_entities = []
                for node in subgraph.nodes():
                    if node in self.kg.nodes():
                        node_data = self.kg.nodes[node]
                        node_type = node_data.get('type', 'unknown')
                        if node_type not in ['Comment', 'Post']:
                            medical_entities.append({
                                'id': node,
                                'type': node_type,
                                'name': node_data.get('name', 'unknown')
                            })
                
                subgraph_info = {
                    'nodes': len(subgraph.nodes()),
                    'edges': len(subgraph.edges()),
                    'medical_entities': medical_entities,
                    'subgraph': subgraph
                }
        
        return {
            'emergency_analysis': emergency_analysis,
            'subgraph_info': subgraph_info,
            'entities': entities,
            'query': query
        }
    
    def get_medical_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Extract medical context from the knowledge graph for a query.
        
        Args:
            query: The medical query
            
        Returns:
            List of medical entities and their information
        """
        result = self.process_query(query)
        if result['subgraph_info']:
            return result['subgraph_info']['medical_entities']
        return []
    
    def is_emergency(self, query: str) -> bool:
        """
        Check if a query indicates an emergency.
        
        Args:
            query: The medical query
            
        Returns:
            True if emergency detected, False otherwise
        """
        emergency_analysis = analyze_medical_urgency(query)
        return emergency_analysis.get('is_emergency', False)
