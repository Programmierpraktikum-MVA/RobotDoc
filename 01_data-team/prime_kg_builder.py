import os
from openai import OpenAI
import argparse
import torch
import pickle

from tqdm import tqdm
import pandas as pd
import networkx as nx

from config import OPENAI_API_KEY, ROOT_DIR


#parser = argparse.ArgumentParser(description='Preprocess PrimeKG')
#parser.add_argument('--prime_kg_dataset', type=str, default='datasets/kg.csv', help='PrimeKG csv path')
#args = parser.parse_args()


class GraphBuilder:
    def __init__(self,
                 prime_kg_path: str,
                 drug_features_path: str,
                 disease_features_path: str,
                 patient_features_path: str,
                 save_path: str = None,
                 filter_edge_or_node_types: list = None
                 ):
        """
        Build a NetwrokX graph based on the PrimeKG csv file.
        Node enbeddings are created using openAI Embeddings Model.
        We use node features provided alongside PrimeKG to enrich the embeddings of drug and diesease nodes.

        Use the save_path parameter to save the generated nx graph as a pickle file.

        Args:
            prime_kg_path: the path to the csv file
            drug_features_path: the path to the drugs features csv file
            disease_features_path: the path to the disease features csv file
            save_path: where to save the generated nx graph and embeddings
            filter_edge_or_node_types: a list of the nodes or edges types to be included in the generated nx graph
        """

        self.embeddings_tensor = None
        self.embeddings_list = []
        self.save_path = save_path
        self.nx_graph = nx.Graph()
        print(self.nx_graph)
        self.model = "text-embedding-ada-002"

        self.node_types = ['gene/protein', 'drug', 'effect/phenotype', 'disease', 'biological_process', 'molecular_function',
                           'cellular_component', 'exposure', 'pathway', 'anatomy', 'symptom', 'patient']

        self.edge_types = ["protein_protein", "drug_protein", "contraindication", "indication", "off_label_use", "drug_drug",
                           "phenotype_protein", "phenotype_phenotype", "disease_phenotype_negative",
                           "disease_phenotype_positive", "disease_protein", "disease_disease", "drug_effect",
                           "bioprocess_bioprocess", "molfunc_molfunc", "cellcomp_cellcomp", "molfunc_protein",
                           "cellcomp_protein", "bioprocess_protein", "exposure_protein", "exposure_disease", "exposure_exposure",
                           "exposure_bioprocess", "exposure_molfunc", "exposure_cellcomp", "pathway_pathway", "pathway_protein",
                           "anatomy_anatomy", "anatomy_protein_present", "anatomy_protein_absent", "symptom_disease", "patient_disease"]

        self.prime_kg_df = pd.read_csv(os.path.join(ROOT_DIR, prime_kg_path))
        print('PrimeKG csv loaded successfully!')

        #if filter_edge_or_node_types is not None:
        #    self.create_sub_df(filter_edge_or_node_types)

        self.drug_features_df = pd.read_csv(os.path.join(ROOT_DIR, drug_features_path), low_memory=False)
        self.disease_features_df = pd.read_csv(os.path.join(ROOT_DIR, disease_features_path), low_memory=False)
        self.patient_features_df = pd.read_csv(os.path.join(ROOT_DIR, patient_features_path), low_memory=False)
        print('Node features csvs loaded successfully!')

        self.generate_nx_graph()
        self.validate_edges()
        self.embed_nodes()

    def create_sub_df(self, types: list, filter_nodes: bool = False):
        df = self.prime_kg_df

        if filter_nodes:
            sub_df = df[df['x_type'].isin(types) and df['y_type'].isin(types)]
        else:
            sub_df = df[df['relation'].isin(types)]

        sub_df.to_csv(os.path.join(ROOT_DIR, 'datasets/prime_kg_{}.csv'.format(", ".join(types))), index=False)
        self.prime_kg_df = sub_df

    def generate_nx_graph(self):

        print('Calling nx.from_pandas_edgelist()...')
        self.nx_graph = nx.from_pandas_edgelist(
            self.prime_kg_df,
            source="x_index",
            target="y_index",
            edge_attr='relation',
            create_using=nx.Graph()
        )
        print('Done!')

        for node in tqdm(list(self.nx_graph.nodes())):
            x_sub_df = self.prime_kg_df.query(f'x_index == {node}')

            # get all edge_types and their respective edges
            #if not x_sub_df.empty:
                #for y_index in list(x_sub_df['y_index']):
                    #rel_type = self.nx_graph.get_edge_data(node, y_index, 0)['relation']

            # create an attributes dictionary for the relevant node
            x_sub_df = x_sub_df[['x_index', 'x_type', 'x_name', 'x_source']].drop_duplicates().rename(
                columns={'x_index': 'index', 'x_type': 'type', 'x_name': 'name', 'x_source': 'source'})

            y_sub_df = self.prime_kg_df.query(f'y_index == {node}')
            y_sub_df = y_sub_df[['y_index', 'y_type', 'y_name', 'y_source']].drop_duplicates().rename(
                columns={'y_index': 'index', 'y_type': 'type', 'y_name': 'name', 'y_source': 'source'})

            node_attributes_dict = pd.concat([x_sub_df, y_sub_df]).drop_duplicates().iloc[0].to_dict()

            # Extract disease and drug features
            features_sub_df = None

            if node_attributes_dict.get('type') == 'disease':
                features_sub_df = self.disease_features_df.query(f'node_index == {node}')
                features_sub_df = features_sub_df[
                    ['mondo_definition', 'umls_description', 'orphanet_definition',
                     'orphanet_prevalence']].drop_duplicates()
                features_sub_df = combine_rows(features_sub_df)

            elif node_attributes_dict.get('type') == 'drug':

                features_sub_df = self.drug_features_df.query(f'node_index == {node}')
                features_sub_df = features_sub_df[['description', 'indication']].drop_duplicates()
                features_sub_df = combine_rows(features_sub_df)

            elif node_attributes_dict.get('type') == 'patient':

                features_sub_df = self.patient_features_df.query(f'x_index == {node}')
                features_sub_df = features_sub_df[['age', 'gender']].drop_duplicates()
                features_sub_df = combine_rows(features_sub_df)


            raw_node_data = f"'name': '{node_attributes_dict.get('name')}', " + f"'type': '{node_attributes_dict.get('type')}'"

            if features_sub_df is not None:
                raw_node_data += ', ' + ', '.join(
                    f"'{col_name}': '{cell_value}'" for col_name, cell_value in features_sub_df.iloc[0].items())

            raw_node_data = raw_node_data.replace("\n", " ")
            self.nx_graph.nodes[node]['raw_data'] = raw_node_data

            self.nx_graph.nodes[node]['type'] = node_attributes_dict.get('type')
            self.nx_graph.nodes[node]['index'] = node_attributes_dict.get('index')
            self.nx_graph.nodes[node]['name'] = node_attributes_dict.get('name')
            self.nx_graph.nodes[node]['source'] = node_attributes_dict.get('source')

            if self.save_path is not None:
                # save graph object to file
                file_name = f"prime_gk_nx_without_embeddings_{len(self.nx_graph.nodes())}.pickle"
                pickle.dump(self.nx_graph, open(os.path.join(ROOT_DIR, self.save_path, file_name), 'wb'))
    
 

    def embed_nodes(self, batch_size=10, model_name='text-embedding-ada-002'):
        """
        Embed a list of texts using the specified model in batches.

        Args:

        model_name (str): The name of the model to use.
        batch_size (int): The number of texts to process in each batch.

        Returns:
        torch.Tensor: A tensor containing the embeddings.
        """
        # Store embeddings
        all_embeddings = []
        client = OpenAI()

        nodes = list(self.nx_graph.nodes())
        for i in tqdm(range(0, len(nodes), batch_size)):
            batch_nodes = nodes[i:i + batch_size]
            batch_data = [self.nx_graph.nodes[node]['raw_data'] for node in batch_nodes]

            try:
                embeddings = client.embeddings.create(input=batch_data, model=model_name).data
                for node, embedding in zip(batch_nodes, embeddings):
                    embedding_tensor = torch.tensor(embedding.embedding)
                    all_embeddings.append(embedding_tensor)
                    self.nx_graph.nodes[node]['embedding'] = embedding_tensor

            except Exception as e:
                print(f"Error processing batch starting at index {i}: {e}")
                continue

        self.embeddings_tensor = torch.cat(all_embeddings, dim=0)

        if self.save_path is not None:
            # save embeddings to file
            embedding_file_name = f'prime_kg_embeddings_tensor_{model_name}_{self.embeddings_tensor.size(0)}'
            torch.save(self.embeddings_tensor, os.path.join(ROOT_DIR, self.save_path, embedding_file_name))

            # save graph object to file
            graph_file_name = f"prime_kg_nx_{model_name}_{len(self.nx_graph.nodes())}"
            pickle.dump(self.nx_graph, open(os.path.join(ROOT_DIR, self.save_path, graph_file_name), 'wb'))

        return self.embeddings_tensor
    

    def validate_edges(self):
        # Convert the DataFrame to a set of tuples for fast lookup
        original_edges = set(zip(self.prime_kg_df['x_index'], self.prime_kg_df['y_index']))

        # Initialize a list to store any edges that don't match
        mismatched_edges = []

        # Iterate over each edge in the graph
        for source, target in self.nx_graph.edges():
            # Convert edge to the same type as DataFrame values for comparison (if needed)
            source = type(self.prime_kg_df['x_index'].iloc[0])(source)
            target = type(self.prime_kg_df['y_index'].iloc[0])(target)

            # Check if the edge exists in the original data
            if (source, target) not in original_edges:
                mismatched_edges.append((source, target))

        # Report the results
        if mismatched_edges:
            print(f"Found {len(mismatched_edges)} mismatched edges.")
            # Optionally print the mismatched edges
            for edge in mismatched_edges:
                print(edge)
        else:
            print("All edges in the graph match the original dataset.")


def combine_rows(df: pd.DataFrame) -> pd.DataFrame:
    combined_row = {}
    for col in df.columns:

        combined_row[col] = df[col].first_valid_index()

        if combined_row[col] is not None:
            combined_row[col] = df[col].loc[combined_row[col]]

    combined_df = pd.DataFrame(combined_row, index=[0])
    return combined_df


# prime_kg_path = 'datasets/kg.csv'
# drug_features_path = 'datasets/drug_features.csv'
# disease_features_path = 'datasets/disease_features.csv'
#
# filter_edge_types = ["indication", "drug_drug", "phenotype_phenotype", "disease_phenotype_positive", "disease_disease", "drug_effect", "phenotype_protein", "disease_protein", "anatomy_anatomy", "anatomy_protein_absent"]
#
# gb = GraphBuilder(prime_kg_path, drug_features_path, disease_features_path, filter_edge_or_node_types=filter_edge_types, save_path='datasets')
