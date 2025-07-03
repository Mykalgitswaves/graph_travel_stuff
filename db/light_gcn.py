from db.driver import MemgraphDriver

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.nn import HeteroConv, Linear
from torch_geometric.data import HeteroData
import pickle
import os


class LightGCN(nn.Module):
    """
    LightGCN implementation for travel recommendation system.
    Simplified version of the original LightGCN paper for heterogeneous graphs.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embedding weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def get_ego_embeddings(self):
        """Get user and item embeddings."""
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def forward(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LightGCN.
        
        Args:
            edge_index: Edge index tensor of shape [2, num_edges]
            
        Returns:
            user_embeddings: Final user embeddings
            item_embeddings: Final item embeddings
        """
        # Get initial embeddings
        ego_embeddings = self.get_ego_embeddings()
        
        # Store embeddings for each layer
        all_embeddings = [ego_embeddings]
        
        # LightGCN propagation
        for layer in range(self.num_layers):
            # Normalize adjacency matrix (symmetric normalization)
            edge_index_norm = self._normalize_adj(edge_index, ego_embeddings.size(0))
            
            # Message passing: E^(l+1) = D^(-1/2) * A * D^(-1/2) * E^(l)
            ego_embeddings = torch.sparse.mm(edge_index_norm, ego_embeddings)
            
            # Apply dropout
            ego_embeddings = F.dropout(ego_embeddings, p=self.dropout, training=self.training)
            
            all_embeddings.append(ego_embeddings)
        
        # Layer combination (sum all layer embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1)
        
        # Split back to users and items
        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.num_users, self.num_items], dim=0
        )
        
        return user_embeddings, item_embeddings
    
    def _normalize_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Normalize adjacency matrix using symmetric normalization.
        
        Args:
            edge_index: Edge index tensor
            num_nodes: Number of nodes
            
        Returns:
            Normalized adjacency matrix as sparse tensor
        """
        # Create adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), device=self.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # Add self-loops
        adj = adj + torch.eye(num_nodes, device=self.device)
        
        # Calculate degree matrix
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        adj_normalized = torch.mm(
            torch.mm(torch.diag(degree_inv_sqrt), adj),
            torch.diag(degree_inv_sqrt)
        )
        
        return adj_normalized.to_sparse()
    
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for user-item pairs.
        
        Args:
            user_ids: User IDs tensor
            item_ids: Item IDs tensor
            
        Returns:
            Prediction scores
        """
        user_embeddings, item_embeddings = self.forward(self.edge_index)
        user_emb = user_embeddings[user_ids]
        item_emb = item_embeddings[item_ids]
        
        # Inner product for prediction
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores


class TravelLightGCN:
    """
    Wrapper class for LightGCN specifically designed for travel recommendations.
    Handles the conversion from heterogeneous graph to user-item bipartite graph.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        self.model = None
        self.user_id_to_idx = {}
        self.item_id_to_idx = {}
        self.idx_to_user_id = {}
        self.idx_to_item_id = {}
        self.user_items = {}
        
    def prepare_data(self, db_driver):
        """
        Prepare data from MemGraph database for LightGCN training.
        
        Args:
            db_driver: MemGraph database driver
        """
        # Query to get traveler-trip relationships
        query = """
        MATCH (traveler:Traveler)-[:TOOK]->(trip:Trip)
        OPTIONAL MATCH (trip)-[:AT_DESTINATION]->(destination:Destination)
        OPTIONAL MATCH (trip)-[:STAYED_IN]->(accommodation:Accommodation)
        OPTIONAL MATCH (trip)-[:TRAVELED_BY]->(transportation:Transportation)
        RETURN 
            traveler.id as traveler_id,
            trip.id as trip_id,
            destination.name as destination,
            accommodation.type as accommodation,
            transportation.type as transportation
        """
        
        result = db_driver.execute_query(query, {})
        
        # Create user-item mappings
        self.user_items = {}
        item_users = {}
        
        for record in result:
            traveler_id = record['traveler_id']
            trip_id = record['trip_id']
            
            # Create composite item ID (trip + destination + accommodation + transportation)
            item_id = f"{trip_id}_{record['destination']}_{record['accommodation']}_{record['transportation']}"
            
            if traveler_id not in self.user_items:
                self.user_items[traveler_id] = set()
            self.user_items[traveler_id].add(item_id)
            
            if item_id not in item_users:
                item_users[item_id] = set()
            item_users[item_id].add(traveler_id)
        
        # Create ID mappings
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_items.keys())}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_users.keys())}
        
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.idx_to_item_id = {idx: item_id for item_id, idx in self.item_id_to_idx.items()}
        
        # Create edge index
        edges = []
        for traveler_id, items in self.user_items.items():
            user_idx = self.user_id_to_idx[traveler_id]
            for item_id in items:
                item_idx = self.item_id_to_idx[item_id]
                # Add bidirectional edges
                edges.append([user_idx, item_idx])
                edges.append([item_idx, user_idx])
        
        self.edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t()
        
        # Initialize model
        self.model = LightGCN(
            num_users=len(self.user_id_to_idx),
            num_items=len(self.item_id_to_idx),
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device
        )
        
        # Store mappings in model for later use
        self.model.user_id_to_idx = self.user_id_to_idx
        self.model.item_id_to_idx = self.item_id_to_idx
        self.model.edge_index = self.edge_index
    
    def train(
        self,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 1024,
        verbose: bool = True
    ):
        """
        Train the LightGCN model.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            verbose: Whether to print training progress
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call prepare_data() first.")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Create training data (positive samples)
        pos_edges = []
        for traveler_id, items in self.user_items.items():
            user_idx = self.user_id_to_idx[traveler_id]
            for item_id in items:
                item_idx = self.item_id_to_idx[item_id]
                pos_edges.append([user_idx, item_idx])
        
        pos_edges = torch.tensor(pos_edges, dtype=torch.long, device=self.device)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            user_embeddings, item_embeddings = self.model.forward(self.edge_index)
            
            # Sample negative edges
            neg_edges = self._sample_negative_edges(pos_edges, len(pos_edges))
            
            # Calculate loss (BPR loss)
            pos_scores = self.model.predict(pos_edges[:, 0], pos_edges[:, 1])
            neg_scores = self.model.predict(neg_edges[:, 0], neg_edges[:, 1])
            
            loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    def _sample_negative_edges(self, pos_edges: torch.Tensor, num_neg: int) -> torch.Tensor:
        """Sample negative edges for training."""
        neg_edges = []
        num_users = len(self.user_id_to_idx)
        num_items = len(self.item_id_to_idx)
        
        for _ in range(num_neg):
            user_idx = torch.randint(0, num_users, (1,)).item()
            item_idx = torch.randint(0, num_items, (1,)).item()
            
            # Make sure it's not a positive edge
            while [user_idx, item_idx] in pos_edges.tolist():
                user_idx = torch.randint(0, num_users, (1,)).item()
                item_idx = torch.randint(0, num_items, (1,)).item()
            
            neg_edges.append([user_idx, item_idx])
        
        return torch.tensor(neg_edges, dtype=torch.long, device=self.device)
    
    def get_recommendations(
        self,
        traveler_id: str,
        top_k: int = 10,
        exclude_visited: bool = True
    ) -> List[Dict]:
        """
        Get travel recommendations for a traveler.
        
        Args:
            traveler_id: Traveler ID
            top_k: Number of recommendations to return
            exclude_visited: Whether to exclude already visited items
            
        Returns:
            List of recommended items with scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if traveler_id not in self.user_id_to_idx:
            raise ValueError(f"Traveler {traveler_id} not found in training data.")
        
        self.model.eval()
        with torch.no_grad():
            user_embeddings, item_embeddings = self.model.forward(self.edge_index)
            
            user_idx = self.user_id_to_idx[traveler_id]
            user_emb = user_embeddings[user_idx].unsqueeze(0)
            
            # Calculate scores for all items
            scores = torch.mm(user_emb, item_embeddings.t()).squeeze()
            
            # Exclude visited items if requested
            if exclude_visited:
                visited_items = self.user_items.get(traveler_id, set())
                for item_id in visited_items:
                    if item_id in self.item_id_to_idx:
                        item_idx = self.item_id_to_idx[item_id]
                        scores[item_idx] = float('-inf')
            
            # Get top-k items
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            
            recommendations = []
            for score, idx in zip(top_scores, top_indices):
                if score > float('-inf'):
                    item_id = self.idx_to_item_id[idx.item()]
                    item_parts = item_id.split('_')
                    recommendations.append({
                        'item_id': item_id,
                        'score': score.item(),
                        'trip_id': item_parts[0],
                        'destination': item_parts[1] if len(item_parts) > 1 else 'Unknown',
                        'accommodation': item_parts[2] if len(item_parts) > 2 else 'Unknown',
                        'transportation': item_parts[3] if len(item_parts) > 3 else 'Unknown'
                    })
            
            return recommendations
    
    def save_model(self, path: str):
        """Save the trained model."""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'num_users': self.model.num_users,
                'num_items': self.model.num_items,
                'embedding_dim': self.model.embedding_dim,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'user_id_to_idx': self.user_id_to_idx,
                'item_id_to_idx': self.item_id_to_idx,
                'user_items': self.user_items,
            }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = 'cpu'):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=device)
        
        instance = cls(
            embedding_dim=checkpoint['embedding_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            device=device
        )
        
        instance.model = LightGCN(
            num_users=checkpoint['num_users'],
            num_items=checkpoint['num_items'],
            embedding_dim=checkpoint['embedding_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            device=device
        )
        
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.user_id_to_idx = checkpoint['user_id_to_idx']
        instance.item_id_to_idx = checkpoint['item_id_to_idx']
        instance.user_items = checkpoint['user_items']
        instance.idx_to_user_id = {idx: user_id for user_id, idx in instance.user_id_to_idx.items()}
        instance.idx_to_item_id = {idx: item_id for item_id, idx in instance.item_id_to_idx.items()}
        
        # Reconstruct edge_index from user_items
        edges = []
        for traveler_id, items in instance.user_items.items():
            user_idx = instance.user_id_to_idx[traveler_id]
            for item_id in items:
                item_idx = instance.item_id_to_idx[item_id]
                # Add bidirectional edges
                edges.append([user_idx, item_idx])
                edges.append([item_idx, user_idx])
        
        instance.edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()

        return instance


# Example usage functions
def train_travel_lightgcn(db_driver, model_path: str = "travel_lightgcn.pth"):
    """
    Train LightGCN model for travel recommendations.
    
    Args:
        db_driver: MemGraph database driver
        model_path: Path to save the trained model
    """
    # Initialize model
    lightgcn = TravelLightGCN(
        embedding_dim=64,
        num_layers=3,
        dropout=0.1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Prepare data
    print("Preparing data...")
    lightgcn.prepare_data(db_driver)
    
    # Train model
    print("Training LightGCN...")
    lightgcn.train(epochs=100, lr=0.001, verbose=True)
    
    # Save model
    lightgcn.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    return lightgcn


def get_travel_recommendations(
    traveler_id: str,
    model_path: str = "travel_lightgcn.pth",
    top_k: int = 10
) -> List[Dict]:
    """
    Get travel recommendations for a traveler using trained LightGCN model.
    
    Args:
        traveler_id: Traveler ID
        model_path: Path to the trained model
        top_k: Number of recommendations to return
        
    Returns:
        List of recommended travel items
    """
    # Load model
    lightgcn = TravelLightGCN.load_model(model_path)
    
    # Get recommendations
    recommendations = lightgcn.get_recommendations(traveler_id, top_k=top_k)
    
    return recommendations


def get_random_traveler_id():
    """Get a random traveler ID from the database."""
    query = """
    MATCH (t:Traveler)
    RETURN t.id
    """
    result = db_driver.execute_query(query, {})
    return result[0]['t.id']

def store_recommendations_for_traveler(traveler_id, recommendations):
    """Store recommendations for a traveler in the database."""
    query = """
        MATCH (t:Traveler {id: $traveler_id})
        WITH t
        UNWIND $recommendations AS rec
        
        MATCH (d:Destination {name: rec.destination})
        MATCH (a:Accommodation {type: rec.accommodation})
        MATCH (tr:Transportation {type: rec.transportation})
        
        // Create recommendation node
        CREATE (r:Recommendation {
            id: rec.rec_id,
            score: rec.score,
            generated_date: datetime(),
            model_version: 'lightgcn_v1',
            confidence: rec.confidence
        })
        
        // Connect traveler to recommendation
        CREATE (t)-[:RECOMMENDED]->(r)
        
        // Connect recommendation to components
        CREATE (r)-[:SUGGESTS_DESTINATION]->(d)
        CREATE (r)-[:SUGGESTS_ACCOMMODATION]->(a)
        CREATE (r)-[:SUGGESTS_TRANSPORTATION]->(tr)
    """
    
    # Prepare recommendations data with all required fields
    recs_data = []
    for i, rec in enumerate(recommendations):
        recs_data.append({
            'rec_id': f"rec_{traveler_id}_{i}_{int(rec['score'] * 1000)}",
            'score': rec['score'],
            'destination': rec['destination'],
            'accommodation': rec['accommodation'],
            'transportation': rec['transportation'],
            'confidence': min(rec['score'] * 1.2, 1.0)
        })
    
    result = db_driver.execute_query(query, {
        'traveler_id': traveler_id,
        'recommendations': recs_data
    })
    print(result)

if __name__ == "__main__":
    db_driver = MemgraphDriver()
    train_travel_lightgcn(db_driver)
    traveler_id = get_random_traveler_id()
    recommendations = get_travel_recommendations(traveler_id, model_path="travel_lightgcn.pth", top_k=10)
    breakpoint()
    print(recommendations)

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. Trip: {rec['trip_id']}")
        print(f"   Destination: {rec['destination']}")
        print(f"   Accommodation: {rec['accommodation']}")
        print(f"   Transportation: {rec['transportation']}")
        print(f"   Score: {rec['score']:.4f}")
        print()

    store_recommendations_for_traveler(traveler_id, recommendations)