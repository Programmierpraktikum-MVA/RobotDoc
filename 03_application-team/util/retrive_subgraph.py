# now we want to retrieve a subgraph for a given dataset

# we have the following input: subject_id (patient_id) und the sysptoms IDS
"""""""""
import networkx as nx
import matplotlib.pyplot as plt
from knowledge_graph import build_graph
from data import get_data


mimic_data = get_data(number_of_rows=10000)

random_row = mimic_data.sample(n=1)
subject_id = random_row['subject_id']
b = random_row['umls_code_list']

print(type(subject_id))

print(type(b))

subject_id = subject_id.values[0]
print("sub -----" + str(subject_id))
print("List : " + str(b))

input_list = []

input_list.append(subject_id)

symptom_list = eval(b.values[0])

for x in symptom_list:
    input_list.append(x)


graph = build_graph()

print("InputList :" + str(input_list) )


sub = graph.subgraph(input_list)

pos = nx.circular_layout(sub)
nx.draw(sub,pos, with_labels=True, node_color='lightblue', edge_color='gray')

#Display the graph
plt.savefig("graph_visualization.png")
"""""""""
import networkx as nx

def get_subgraph(graph, symptom):
    # Initialize an empty subgraph
    subgraph = nx.Graph()

    # Add the given symptom to the subgraph
    subgraph.add_node(symptom)

    # Iterate through the neighbors of the symptom node in the original graph
    for neighbor in graph.neighbors(symptom):
        # Add the neighbor (related disease) to the subgraph
        subgraph.add_node(neighbor)

        # Add an edge between the symptom and its related disease in the subgraph
        subgraph.add_edge(symptom, neighbor)

    return subgraph


