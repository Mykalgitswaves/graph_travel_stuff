from db.driver import MemgraphDriver
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def run_pca_on_all_feature_embeddings():
    db = MemgraphDriver()
    query = """
        MATCH (t:Traveler) where t.feature_embeddings is not NULL
        return t.feature_embeddings as f_embeddings
    """
    result = db.execute_query(query)
    embeddings = []

    for record in result:
        embeddings.append(record['f_embeddings'])
    
    features = np.array(embeddings)
    pca = PCA(n_components=3)
    breakpoint()
    pca.fit(features)

    # This gives you the loadings: how important each feature is for each principal component
    loadings = pca.components_

    # For example, you can look at the first principal component
    print("First principal component feature loadings:")
    print(loadings[0])

    # If you want to know how much variance each component explains
    print("Explained variance ratio:")
    print(pca.explained_variance_ratio_)

    feature_names = [f"feature_{i}" for i in range(features.shape[1])]

    # First component
    first_pc = loadings[0]

    # Create a DataFrame to make it pretty
    importance = pd.DataFrame({
        "feature": feature_names,
        "loading": first_pc
    })

    # Sort by absolute importance
    importance["abs_loading"] = importance["loading"].abs()
    importance = importance.sort_values("abs_loading", ascending=False)

    print(importance)
    breakpoint()

if __name__ == "__main__":
    run_pca_on_all_feature_embeddings()
