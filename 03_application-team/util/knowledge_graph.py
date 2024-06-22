import networkx as nx
import matplotlib.pyplot as plt
from util.data import get_data
from util.extract_features import extract_input_nodes_as_tuples, extract_output_nodes_as_tuples
from util.define_relationships import define_desease_symtoms_between_nodes, define_patient_symptoms_relations, define_patients_desease_relations
from util.retrive_subgraph import get_subgraph


mimic_data = get_data(number_of_rows= 1000)

def define_nodes(input_nodes, output_nodes, graph):

    for node in input_nodes:

        graph.add_node( node[0] , uml_description = node[1] )
    

    for node in output_nodes:

        graph.add_node( node[0], icd_description = node[1] )



def build_graph(  threshhold_desease_symtoms= 100, threshhold_patient_symtoms=1, threshhold_patieent_desease=1 ):
    graph = nx.Graph()

    input_nodes = extract_input_nodes_as_tuples(mimic_data)

    output_nodes = extract_output_nodes_as_tuples(mimic_data)

    #patient_nodes = define_patient_tuples(mimic_data)

    define_nodes(input_nodes, output_nodes, graph)


    define_desease_symtoms_between_nodes(mimic_data, graph)

    #define_patient_symptoms_relations(mimic_data, graph, threshhold_patient_symtoms)

    #define_patients_desease_relations(mimic_data, graph, threshhold_patieent_desease)


    return graph

"""""""""""""""""""""
graph = build_graph()
fig, ax = plt.subplots(figsize=(24, 24))
pos = nx.circular_layout(graph)
nx.draw(graph,pos, with_labels=True, node_color='blue', edge_color='red', width=2.0)

#BİR KISMININ BAŞINDA BOŞLUK VAR, ANCA ÖYLE ÇALIŞIYOR MESELA ' nausea'
desired_symptom = ' nausea'

# Retrieve the subgraph for the desired symptom
subgraph = get_subgraph(graph, desired_symptom)

# Display the graph
plt.savefig("graph_visualization.png")
plt.savefig("static/img/graph_visualization.png")

pos = nx.spring_layout(subgraph)  # You can choose a layout algorithm that suits your needs
nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray')

# Display the subgraph
plt.savefig("subgraph_visualization.png")
plt.savefig("static/img/subgraph_visualization.png")
plt.show()
"""""""""""""""""""""








