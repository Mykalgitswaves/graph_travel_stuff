from db.driver import MemgraphDriver
import torch
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA


def create_destination_encoder():
    db = MemgraphDriver()
    query = """
        MATCH (dest:Destination)
        RETURN dest.name as destination
    """
    result = db.execute_query(query, {})
    all_destinations = [record['destination'] for record in result]
    destination_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    dest_encoder = destination_encoder.fit(np.array([all_destinations]).reshape(-1, 1))
    return dest_encoder
    
def create_accomodations_encoder():
    db = MemgraphDriver()
    query = """
        match (accom:Accommodation) return accom.type as accommodation
    """
    result = db.execute_query(query, {})
    all_accoms = (record['accommodation'] for record in result)
    destination_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    accom_encoder = destination_encoder.fit(np.array(all_accoms).reshape(-1, 1))
    return accom_encoder

def create_transportation_encoder():
    db = MemgraphDriver()
    query = """
        MATCH (transport:Transportation)
        RETURN transport.type as transport
    """
    result = db.execute_query(query, {})
    all_transports = [record['transport'] for record in result]
    destination_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    transport_encoder = destination_encoder.fit(np.array([all_transports]).reshape(-1, 1))
    return transport_encoder

def create_traveler_features(user_id: str):
    db = MemgraphDriver()
    query = """
    MATCH (t:Traveler {id: $user_id})
    OPTIONAL MATCH (t)-[tr:TOOK]-(trip:Trip)
    OPTIONAL MATCH (trip)-[:AT_DESTINATION]->(destination:Destination)
    OPTIONAL MATCH (trip)-[:STAYED_IN]->(accommodation:Accommodation)
    OPTIONAL MATCH (trip)-[:TRAVELED_BY]->(transport:Transportation)
    RETURN t as traveler, trip, count(trip) as trips_count, destination, accommodation, transport
    """
    result = db.execute_query(query, {"user_id": user_id})
    
    t_features = {
        'avg_trip_duration': np.mean([int(record['trip'].get('duration')) for record in result]),
        'total_trips': np.sum([record['trips_count'] | 0 for record in result]),
        'destinations_visited': [record['destination']['name'] for record in result],
        'transportation_used': [record['transport']['type'] for record in result],
        'accommodation_types': [record['accommodation']['type'] for record in result],
        'avg_trip_cost': np.median([int(record['trip'].get('transportationCost', 0)) + int(record['trip'].get('accommodationCost', 0)) for record in result]),
        'seasonality': [int(record['trip'].get('startDate').month) for record in result],
        'traveler_ages': np.median([int(record['traveler']['age']) for record in result]),
        'traveler_genders': [record['traveler']['gender'] for record in result]
    }
    
    return t_features

# Encode features from our traveler and our reference features.
def encode_categorical_features(features, destination_encoder, transport_encoder, accommodation_encoder):
    #destinations
    dest_array = np.array(features['destinations_visited']).reshape(-1, 1)
    dest_encoded = destination_encoder.transform(dest_array)
    dest_vector = np.sum(dest_encoded, axis=0)  # Sum or mean depending on preference

    # transportation
    trans_array = np.array(features['transportation_used']).reshape(-1, 1)
    trans_encoded = transport_encoder.transform(trans_array)
    trans_vector = np.sum(trans_encoded, axis=0)

    # Accoms
    acc_array = np.array(features['accommodation_types']).reshape(-1, 1)
    acc_encoded = accommodation_encoder.transform(acc_array)
    acc_vector = np.sum(acc_encoded, axis=0)

    combined_vector = np.concatenate([dest_vector, trans_vector, acc_vector])

    return combined_vector


def generate_feature_embeddings_for_user(user_id, destination_encoder, transportation_encoder, accommodations_encoder, pca=None):
    """
    Generate feature embeddings for a user based on their travel preferences and history.
    
    Parameters:
    -----------
    user_id : str or int
        Unique identifier for the user
    destination_encoder : encoder object
        Fitted encoder for destination features
    transportation_encoder : encoder object
        Fitted encoder for transportation features
    accommodations_encoder : encoder object
        Fitted encoder for accommodations features
    pca : PCA object, optional
        Fitted PCA for dimensionality reduction
        
    Returns:
    --------
    numpy.ndarray
        Feature embedding vector for the user
    """
    # Get user features
    traveler_features = create_traveler_features(user_id=user_id)
    
    # Get categorical features
    cat_vector = encode_categorical_features(traveler_features, 
                                             destination_encoder, 
                                             transportation_encoder, 
                                             accommodations_encoder)
    
    # MAKE EVERYTHING A DICT
    numerical_features = {
        'avg_trip_duration': traveler_features['avg_trip_duration'],
        'avg_trip_cost': traveler_features['avg_trip_cost'],
        'total_trips': traveler_features['total_trips'],
        'age': traveler_features['traveler_ages']
    }
    
    # Create numerical vector and normalize
    numerical_vector = np.array([
        numerical_features['avg_trip_duration'],
        numerical_features['avg_trip_cost'],
        numerical_features['total_trips'],
        numerical_features['age']
    ]).reshape(-1, 1)
    
    scaler = MinMaxScaler()
    normalized_numerical_vector = scaler.fit_transform(numerical_vector).flatten()
    
    weights = {
        'avg_trip_duration': 1.2,
        'avg_trip_cost': 1.2,
        'total_trips': 1.0,
        'age': 1.3 # I think age is more important. 
    }
    
    weighted_numerical_vector = np.array([
        normalized_numerical_vector[0] * weights['avg_trip_duration'],
        normalized_numerical_vector[1] * weights['avg_trip_cost'],
        normalized_numerical_vector[2] * weights['total_trips'],
        normalized_numerical_vector[3] * weights['age']
    ])
    
    gender = traveler_features['traveler_genders'][0]
    gender_encoder = OneHotEncoder(sparse_output=False)
    gender_one_hot = gender_encoder.fit_transform([[gender]])
    
    combined_vector = np.concatenate([cat_vector, weighted_numerical_vector, gender_one_hot.flatten()])
    
    if pca is not None:
        # Reshape to 2D for PCA
        combined_vector_2d = combined_vector.reshape(1, -1)
        reduced_vector = pca.transform(combined_vector_2d)
        # Add small noise to prevent exact matches - this is gippitied and doesn't seem to work
        noise = np.random.normal(0, 0.01, size=reduced_vector.shape)
        return (reduced_vector + noise).flatten()
    
    return combined_vector

def prepare_all_embeddings(user_ids, destination_encoder, transportation_encoder, accommodations_encoder):
    all_embeddings = []
    
    # First, collect all embeddings
    for user_id in user_ids:
        try:
            embeddings = generate_feature_embeddings_for_user(
                user_id=user_id,
                destination_encoder=destination_encoder,
                transportation_encoder=transportation_encoder,
                accommodations_encoder=accommodations_encoder
            )
            all_embeddings.append(embeddings)
        except KeyError as e:
            print(f"Error while generating embeddings for user {user_id}: {str(e)}")
            continue
    
    # Convert to numpy array
    all_embeddings = np.array(all_embeddings)
    
    # Fit PCA on all embeddings
    pca = PCA(n_components=6)
    pca.fit(all_embeddings)
    
    return pca

def find_similar_travelers(user_id, n_similar=5):
    db = MemgraphDriver()
    
    # Get all users' embeddings
    query = """
        MATCH (t:Traveler)
        WHERE t.feature_embeddings IS NOT NULL
        RETURN t.id as id, t.feature_embeddings as embeddings
    """
    results = db.execute_query(query, {})
    
    # Get target user's embedding
    target_embedding = None
    other_embeddings = {}
    
    for record in results:
        id = record['id']
        embedding = record['embeddings']
        
        if id == user_id:
            target_embedding = np.array(embedding)
        else:
            other_embeddings[id] = np.array(embedding)
    
    if target_embedding is None:
        return []
    
    # Calculate raw distances and similarities
    raw_similarities = {}
    for other_id, other_embedding in other_embeddings.items():
        try:
            # Calculate cosine similarity
            dot_product = np.dot(target_embedding, other_embedding)
            target_norm = np.linalg.norm(target_embedding)
            other_norm = np.linalg.norm(other_embedding)
            
            if target_norm == 0 or other_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (target_norm * other_norm)
            
            raw_similarities[other_id] = similarity
        except Exception as e:
            print(f"Error calculating similarity with user {other_id}: {str(e)}")
    
    # Apply a more moderate contrast enhancement
    enhanced_similarities = {}
    if raw_similarities:
        # Get min and max for normalization
        min_sim = min(raw_similarities.values())
        max_sim = max(raw_similarities.values())
        range_sim = max_sim - min_sim
        
        if range_sim > 0.001:  # Avoid division by zero
            for other_id, sim in raw_similarities.items():
                # Normalize to 0-1 range
                normalized = (sim - min_sim) / range_sim
                
                # Apply a more moderate enhancement (sqrt is more gentle than square)
                # This will spread out the middle values without pushing everything to extremes
                enhanced = normalized ** 1.5  # Use 1.5 power instead of 2
                
                enhanced_similarities[other_id] = enhanced
        else:
            # If all similarities are nearly identical, add small random variations
            for other_id, sim in raw_similarities.items():
                enhanced_similarities[other_id] = sim + np.random.uniform(-0.05, 0.05)
    
    # Convert to list and sort
    similarities = [(id, sim) for id, sim in enhanced_similarities.items()]
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Format similarity scores to 4 decimal places for display
    formatted_similarities = [(id, round(sim, 4)) for id, sim in similarities[:n_similar]]
    
    return formatted_similarities

def store_feature_embeddings_on_user_node(user_id:str, feature_embeddings:np.array):
    if not feature_embeddings.any(): 
        raise KeyError("no feature embeddings provided")
    
    # Convert to list for storage
    embeddings_list = feature_embeddings.tolist()
    
    db = MemgraphDriver()
    query = """
        MATCH (t:Traveler {id: $user_id})
        SET t.feature_embeddings = $feature_embeddings
    """
    result = db.execute_query(query, {"user_id": user_id, "feature_embeddings": embeddings_list})
    if result:
        print(f"user {user_id} d_embeddings set 🫡")


if __name__ == "__main__":
    destination_encoder = create_destination_encoder()
    accommodations_encoder = create_accomodations_encoder()
    transportation_encoder = create_transportation_encoder()

    user_ids = [
        "b6a229b9-8b77-46fb-9612-33f5ddfaa2e5",
        "57be7d3a-8aac-46bf-ba1b-89ab280a489e",
        "4019d6ff-d5f1-4d99-a6a1-bec0644afc11",
        "92e6330d-ad13-4451-8e90-b99d91764d38",
        "87f9c9fe-cedf-4292-95b7-3bef7f822d43",
        "f68fe24e-91a2-42a5-9fa2-c0ca66178a16",
        "0fa8d56e-ee1f-4b39-a0fe-d2cc8c82b56b",
        "bc3b84c2-bd09-4098-b94d-74f1dd71a514",
        "4af7b153-5d2c-48b7-9069-dd95b210da8e",
        "a4aef068-d0a1-4ecd-98f0-978c75739de8",
    ]

    # First, prepare PCA on all embeddings
    pca = prepare_all_embeddings(
        user_ids=user_ids,
        destination_encoder=destination_encoder,
        transportation_encoder=transportation_encoder,
        accommodations_encoder=accommodations_encoder
    )

    all_embeddings = []
    processed_user_ids = []

    # Then, generate and store reduced embeddings for all users
    for id in user_ids:
        try:
            feature_embeddings = generate_feature_embeddings_for_user(
                user_id=id, 
                destination_encoder=destination_encoder, 
                accommodations_encoder=accommodations_encoder, 
                transportation_encoder=transportation_encoder,
                pca=pca
            )
            store_feature_embeddings_on_user_node(user_id=id, feature_embeddings=feature_embeddings)
            # feature_embeddings = generate_feature_embeddings_for_user(
            #     user_id=id, 
            #     destination_encoder=destination_encoder, 
            #     transportation_encoder=transportation_encoder, 
            #     accomodations_encoder=accomodations_encoder
            # )
            # all_embeddings.append(feature_embeddings)
            # processed_user_ids.append(id)
        except KeyError as e:
            print(f"Error while setting feature {str(e)}")
    
    #  # Determine optimal number of components
    # if len(all_embeddings) >= 3:  # Need at least 3 points for meaningful PCA
    #     optimal_components = determine_optimal_pca_components(all_embeddings)
    #     pca = PCA(n_components=optimal_components)
    # else:
    #     print("Not enough users for PCA analysis, using default components")
    #     pca = PCA(n_components=min(10, len(all_embeddings)))

    # Then, find similar travelers for each user
    for id in user_ids:
        similar_travelers = find_similar_travelers(id)
        print(f"\nSimilar travelers for user {id}:")
        for similar_id, similarity in similar_travelers:
            print(f"User {similar_id}: similarity score {similarity:.4f}")



# def determine_optimal_pca_components(all_embeddings):
#     """Analyze and determine the optimal number of PCA components"""
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
#     import numpy as np
    
#     # Convert to numpy array if it's not already
#     embeddings_array = np.array(all_embeddings)
    
#     # Get the maximum possible components (min of n_samples and n_features)
#     max_components = min(embeddings_array.shape[0], embeddings_array.shape[1])
    
#     # If we have very few samples, we might need to limit further
#     if max_components <= 2:
#         print(f"Only {max_components} components possible. Using all available.")
#         return max_components
    
#     # Create a PCA object with maximum possible components
#     pca = PCA(n_components=max_components)
#     pca.fit(embeddings_array)
    
#     # Get explained variance ratio
#     explained_variance = pca.explained_variance_ratio_
#     cumulative_variance = np.cumsum(explained_variance)
    
#     # Print variance information
#     print(f"Total features: {embeddings_array.shape[1]}")
#     print(f"Maximum possible components: {max_components}")
#     print("\nExplained variance by components:")
#     for i, variance in enumerate(explained_variance[:10]):
#         print(f"Component {i+1}: {variance:.4f} ({cumulative_variance[i]:.4f} cumulative)")
    
#     # Method 1: Find components needed for 95% variance
#     components_95 = np.argmax(cumulative_variance >= 0.95) + 1
#     print(f"\nComponents needed for 95% variance: {components_95}")
    
#     # Method 2: Find the "elbow" in the scree plot
#     # Calculate second derivative to find the point of maximum curvature
#     second_derivative = np.diff(np.diff(explained_variance, n=1), n=1)
#     elbow_index = np.argmax(np.abs(second_derivative[:max(5, len(second_derivative)-1)])) + 2
#     print(f"Elbow point (maximum curvature): Component {elbow_index}")
    
#     # Create scree plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-', linewidth=2)
#     plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', linewidth=2)
#     plt.axhline(y=0.95, color='g', linestyle='--')
#     plt.axvline(x=components_95, color='g', linestyle='--')
#     plt.axvline(x=elbow_index, color='r', linestyle='--')
#     plt.title('Scree Plot')
#     plt.xlabel('Principal Component')
#     plt.ylabel('Explained Variance Ratio')
#     plt.legend(['Individual Variance', 'Cumulative Variance', '95% Threshold', 
#                 f'95% Components ({components_95})', f'Elbow Point ({elbow_index})'])
#     plt.grid(True)
#     plt.tight_layout()
    
#     # Save the plot
#     plt.savefig('pca_components_analysis.png')
#     print("Scree plot saved as 'pca_components_analysis.png'")
    
#     # Recommend a number of components
#     recommended = min(components_95, max(3, elbow_index))
#     print(f"\nRecommended number of components: {recommended}")
    
#     return recommended