import pickle

from tqdm import tqdm


class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char):
        # the character stored in this node
        self.char = str(char)

        # whether this can be the end of a word
        self.is_end = False

        # a counter indicating how many times a word is inserted
        # (if this node's is_end is True)
        self.counter = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}

        # stores the edges to the arg1 key
        # a list of trippels ['arg1', 'arg2', 'rel']
        self.edges = []

        # stores a list of the embeddings of the node's neighbors
        self.embedding = []

        # stores the unique node indices (in case they exist in the graph) for each edge
        # for easier subgraph construction later on
        self.edges_indices = []


class Trie(object):

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode("")

        self.seen_entities_list = []  # an optional parameter to help avoid cycles when querying the trie

    def insert_dataset(self, d_set, embs):
        for index, data in tqdm(enumerate(d_set), total=len(d_set)):
            self.insert(data['arg1'], [data['arg1'], data['arg2'], data['rel']], embs[index])

    def insert_knowledge_graph(self, kg, node_attribute, store_node_index=False, kg_embeddings=None):
        """Insert a KG into the trie"""
        """node_attribute is the attribute that is going to be inserted to the trie 
            (node's name, description, identifier...."""

        """kg_embeddings can be use to pass an external embeddings file, in case the information is not stored 
            in the graph. In that case the current node's embedding can be accessed via kg_embeddings[index]"""
        
        #Knoten und ihre Attribute anzeigen
        #print("Knoten und ihre tensor shit:")
        #count = 0
        #for node in kg.nodes(data=True):
        #    if count < 500:
        #        embedding = node[1].get('embedding')
        #        print(f"Knoten {node}: {embedding.size()}")
        #        print(f"Knoten {node}: {embedding.dim()}")
        #        count += 1



    


        for index, (u, v, data) in enumerate(kg.edges(data=True)):
            source = kg.nodes[u]
            target = kg.nodes[v]
            source_attribute = "_".join(source[node_attribute].split())
            source_name = "_".join(source['name'].split())
            target_name = "_".join(target['name'].split())
            
            #if target['index'] >= len(kg_embeddings):
            #    continue


            if store_node_index:
                self.insert(source_attribute, [source_name, target_name, data['relation']], target['embedding'],
                            source['index'], target['index'])
            else:
                self.insert(source_attribute, [source_name, target_name, data['relation']], target['embedding'])

    def insert(self, word, edge, target_embedding, source_index=None, target_index=None):
        """Insert a word into the trie"""
        node = self.root

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for char in word:
            char = str(char)
            if str(char) in node.children:
                node = node.children[str(char)]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(char)
                node.children[str(char)] = new_node
                node = new_node

        # Mark the end of a word
        node.is_end = True

        # Insert edge to edges list
        node.edges.append(edge)

        node.embedding.append(target_embedding)

        # Increment the counter to indicate that we see this word once more
        node.counter += 1

        if source_index is not None and target_index is not None:
            node.edges_indices.append([source_index, target_index])

    def query(self, x, avoid_cycles=False):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted

        Set avoid_cycles=True for avoiding cycles when traversing through a graph.
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        node = self.root
   

        # Check if the prefix is in the trie
        for char in x:
            char = str(char)
            if str(char) in node.children:
                node = node.children[str(char)]
            else:
                # cannot found the prefix, return empty list
                return [], [], []

        # return only edges and embeddings of new triples (avoid cycles in graph)
        edges, embeddings, node_indices = [], [], []

        if avoid_cycles:

            for i in range(len(node.edges)):

                if node.edges[i][1] not in self.seen_entities_list:
                    
                    edges.append(node.edges[i])
                    embeddings.append(node.embedding[i])
                    
                    #print(f"Knoten size {node}: {node.embedding[i].size()}")
                    #print(f"Knoten dim {node}: {node.embedding[i].dim()}")
                    node_indices.append(node.edges_indices[i])
            self.seen_entities_list.append(x)
        else:
            edges = node.edges
            embeddings = node.embedding
            node_indices = node.edges_indices
        #print("amount of edges:" + str(len(edges)))
        #print("amount of embeddings:" + str(len(embeddings)))
        #print("amount of node indices:" + str(len(node_indices)))
  
        
        return edges, embeddings, node_indices

    def save_trie(self, path):
        with open(path + '/trie.pickle', 'wb') as f:
            pickle.dump(self, f)
