from db.driver import MemgraphDriver
import torch
from torch_geometric.nn import MetaPath2Vec
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import base64
import numpy as np

# This script now uses metapath2vec++ style: all walks are unidirectional along the specified meta-paths.

traveler_destination_traveler = [
    ('Traveler', 'TOOK', 'Trip'),
    ('Trip', 'AT_DESTINATION', 'Destination'),
]

traveler_accommodation_traveler = [
    ('Traveler', 'TOOK', 'Trip'),
    ('Trip', 'STAYED_IN', 'Accommodation'),
]

traveler_transportation_traveler = [
    ('Traveler', 'TOOK', 'Trip'),
    ('Trip', 'TRAVELED_BY', 'Transportation'),
]


# Convert your Memgraph data to a PyTorch Geometric format
def create_pytorch_geometric_graph():
    db = MemgraphDriver()
    query = """
        MATCH (traveler:Traveler)
        OPTIONAL MATCH (traveler)-[r1:TOOK]->(trip:Trip)
        OPTIONAL MATCH (trip)-[r2:AT_DESTINATION]->(destination:Destination)
        OPTIONAL MATCH (trip)-[r3:STAYED_IN]->(accommodation:Accommodation)
        OPTIONAL MATCH (trip)-[r4:TRAVELED_BY]->(transportation:Transportation)
        RETURN traveler, r1, trip, r2, destination, r3, accommodation, r4, transportation
    """
    result = db.execute_query(query, {})
    
    # Create a NetworkX graph
    G = nx.MultiDiGraph()
    
    # Create mappings from IDs to indices
    global traveler_id_to_idx
    traveler_id_to_idx = {}

    node_to_idx = {}
    idx = 0

    # Add nodes and edges from query results
    for record in result:
       # Extract nodes from the record
        traveler = record.get('traveler')
        trip = record.get('trip')
        destination = record.get('destination')
        accommodation = record.get('accommodation')
        transportation = record.get('transportation')
        
        # Add nodes with their properties
        if traveler:
            traveler_uuid = traveler._properties.get('id')
            props = dict(traveler._properties)
            # We don't want to include the embedding or feature_embeddings in the node properties.
            if 'embedding' in props:
                del props['embedding']
            if 'feature_embeddings' in props:
                del props['feature_embeddings']

            G.add_node(traveler.id, type='Traveler', **props)
            if traveler.id not in node_to_idx:
                node_to_idx[traveler.id] = idx

                if traveler_uuid:
                    traveler_id_to_idx[traveler_uuid] = idx
                idx += 1
        
        if trip:
            G.add_node(trip.id, type='Trip', **trip._properties)
        
        if destination:
            G.add_node(destination.id, type='Destination', **destination._properties)
        
        if accommodation:
            props = dict(accommodation._properties)
            if 'type' in props:
                del props['type']
            G.add_node(accommodation.id, type='Accommodation', **props)
        
        if transportation:
            props = dict(transportation._properties)
            if 'type' in props:
                del props['type']
            G.add_node(transportation.id, type='Transportation', **props)
        
        # Add only forward edges (unidirectional)
        if traveler and trip:
            G.add_edge(traveler.id, trip.id, type='TOOK')
        
        if trip and destination:
            G.add_edge(trip.id, destination.id, type='AT_DESTINATION')
        
        if trip and accommodation:
            G.add_edge(trip.id, accommodation.id, type='STAYED_IN')
        
        if trip and transportation:
            G.add_edge(trip.id, transportation.id, type='TRAVELED_BY')
    
    

    # Convert to PyTorch Geometric format
    edge_types = [
        ('Traveler', 'TOOK', 'Trip'),
        ('Trip', 'AT_DESTINATION', 'Destination'),
        ('Trip', 'STAYED_IN', 'Accommodation'),
        ('Trip', 'TRAVELED_BY', 'Transportation'),
    ]
    edge_index_dict = {}
    
    for node in G.nodes():
        node_to_idx[node] = idx
        idx += 1

    # Create edge_index_dict
    for source_type, edge_type, target_type in edge_types:
        source_nodes = []
        target_nodes = []
        
        # Find all edges of this type
        for u, v, data in G.edges(data=True):
            if (G.nodes[u].get('type') == source_type and 
                G.nodes[v].get('type') == target_type and 
                data.get('type') == edge_type):
                source_nodes.append(node_to_idx[u])
                target_nodes.append(node_to_idx[v])
        
        # Only add to dict if we found edges of this type
        if source_nodes:
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            edge_index_dict[(source_type, edge_type, target_type)] = edge_index

    print(f"Number of travelers in DB: ...")  # Query your DB for this
    print(f"Number of travelers in mapping: {len(traveler_id_to_idx)}")
    print(f"Traveler IDs in mapping: {list(traveler_id_to_idx.keys())[:10]}")  # Show a sample

    return edge_index_dict

def get_traveler_index(traveler_id, traveler_id_to_idx):
    """
    Get the index of a traveler in the embedding matrix based on their ID.
    
    Parameters:
    -----------
    traveler_id : str
        The UUID or ID of the traveler in the database
        
    Returns:
    --------
    int
        The index of the traveler in the embedding matrix
    """
    # You need to maintain a mapping from traveler IDs to their indices
    # This could be created during graph construction
    traveler_id_to_idx  # This should be created during graph construction
    if traveler_id not in traveler_id_to_idx:
        raise KeyError(f"Traveler with ID {traveler_id} not found in the graph")
    
    return traveler_id_to_idx[traveler_id]

def get_traveler_embedding(traveler_id, embeddings_dict, traveler_id_to_idx):
    # Get the index of the traveler in your graph
    traveler_idx = get_traveler_index(traveler_id, traveler_id_to_idx)
    
    # Combine embeddings from different meta-paths
    combined_embedding = torch.cat([
        embeddings_dict["metapath_0"][traveler_idx] * 0.4, #destination
        embeddings_dict["metapath_1"][traveler_idx] * 0.3, #accommodation
        embeddings_dict["metapath_2"][traveler_idx] * 0.3 #transportation
    ])
    
    return combined_embedding.numpy().reshape(1, -1)

def find_similar_travelers(traveler_id, embeddings_dict, traveler_id_to_idx, top_k=5):
    target_embedding = get_traveler_embedding(traveler_id, embeddings_dict, traveler_id_to_idx)
    
    similarities = []
    for other_id in traveler_id_to_idx.keys():
        if other_id == traveler_id:
            continue
        
        other_embedding = get_traveler_embedding(other_id, embeddings_dict, traveler_id_to_idx)
        similarity = cosine_similarity(target_embedding, other_embedding)[0][0]
        similarities.append((other_id, similarity))
    
    # Sort by similarity and return top k
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

def create_embeddings_for_all_travelers_and_save_to_file():
# Create the graph
    edge_index_dict = create_pytorch_geometric_graph()

    # Define meta-paths
    metapaths = [
        traveler_destination_traveler,
        traveler_accommodation_traveler,
        traveler_transportation_traveler
    ]

    # Create and train models for each meta-path
    embeddings_dict = {}
    for i, metapath in enumerate(metapaths):
        walk_length = len(metapath)
        context_size = walk_length + 1  # or less, but not more
        model = MetaPath2Vec(
            edge_index_dict=edge_index_dict,
            embedding_dim=64,
            metapath=metapath,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=5,
            num_negative_samples=5,
            sparse=True
        )
        
        # Training code
        loader = model.loader(batch_size=128, shuffle=True)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        
        # Train for several epochs
        for epoch in range(10):
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw, neg_rw)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Metapath {i}, Epoch {epoch}, Loss: {total_loss/len(loader)}")
        
        # Store embeddings
        embeddings_dict[f"metapath_{i}"] = model.embedding.weight.data
        save_data = {
            'embeddings': embeddings_dict,
            'traveler_id_to_idx': traveler_id_to_idx
        }
        # Save embeddings to a file
        with open('traveler_embeddings.pkl', 'wb') as f:
            pickle.dump(save_data, f)

def store_feature_keys_on_traveler_nodes(traveler_id, embeddings_dict, traveler_id_to_idx):
    embedding_for_traveler = get_traveler_embedding(traveler_id, embeddings_dict, traveler_id_to_idx)
    base64_embedding = base64.b64encode(embedding_for_traveler.tobytes()).decode('utf-8')

    db = MemgraphDriver()
    query = """
        MATCH (traveler:Traveler {id: $traveler_id})
        SET traveler.b64_graph_embeddings = $graph_embeddings
        RETURN traveler
    """
    result = db.execute_query(query, {'traveler_id': traveler_id, 'graph_embeddings': base64_embedding})
    
    return result

# Retrieving
def decode_embedding(b64_str:str, dtype_str:str, shape_str:str):
    dtype = np.dtype(dtype_str)
    shape = tuple(int(x) for x in shape_str.strip('()').split(',') if x)
    
    bytes_data = base64.b64decode(b64_str)
    array = np.frombuffer(bytes_data, dtype=dtype)
    
    if len(shape) > 1:
        array = array.reshape(shape)
    
    return array


if __name__ == "__main__":
    # create_embeddings_for_all_travelers_and_save_to_file()

    # Load embeddings from a file
    with open('traveler_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)

        highest_similarity = {
            'traveler_id': None,
            'similarity': 0,
            'similar_to_id': None,
        }

        for traveler_id in data['traveler_id_to_idx'].keys():
            store_feature_keys_on_traveler_nodes(traveler_id, data['embeddings'], data['traveler_id_to_idx'])

        # Find similar travelers
        for traveler_id in data['traveler_id_to_idx'].keys():
            most_similar_traveler_for_traveler = find_similar_travelers(traveler_id, data['embeddings'], data['traveler_id_to_idx'], top_k=1)[0]
            highest_similarity_for_traveler_score = most_similar_traveler_for_traveler[1]
            if highest_similarity_for_traveler_score > highest_similarity['similarity']:
                highest_similarity['similarity'] = highest_similarity_for_traveler_score
                highest_similarity['similar_to_id'] = most_similar_traveler_for_traveler[0]
                highest_similarity['traveler_id'] = traveler_id

        breakpoint()