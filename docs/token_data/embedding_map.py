from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import numpy as np

model = SentenceTransformer("google/embeddinggemma-300m")

query = "What are popular tourist attractions in Paris?"

documents = [
    "The Eiffel Tower is one of the most iconic landmarks in Paris and attracts millions of visitors.",
    "The Louvre Museum houses thousands of artworks including the famous Mona Lisa painting.",
    "Tokyo is known for its blend of traditional temples and modern skyscrapers.",
    "Notre-Dame Cathedral is a masterpiece of Gothic architecture located on Île de la Cité.",
    "The Great Wall of China stretches over 13,000 miles across northern China.",
    "Montmartre is a charming hilltop neighborhood known for its artistic history and the Sacré-Cœur Basilica.",
    "New York City's Statue of Liberty symbolizes freedom and democracy.",
    "The Champs-Élysées is a famous avenue lined with shops, cafes, and theaters.",
    "The Amazon rainforest is home to incredible biodiversity and indigenous communities.",
    "The Palace of Versailles showcases the grandeur of French royal history.",
    "Sydney Opera House is an architectural marvel on the Australian waterfront.",
    "Arc de Triomphe honors those who fought for France and offers panoramic city views.",
    "The Grand Canyon offers breathtaking views of layered rock formations.",
    "Seine River cruises provide a romantic way to see Paris's beautiful bridges and landmarks."
]

query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)

all_embeddings = np.vstack([query_embeddings.reshape(1, -1), document_embeddings])
tsne = TSNE(n_components=3, random_state=42, perplexity=5)
embeddings_3d = tsne.fit_transform(all_embeddings)

similarities = model.similarity(query_embeddings, document_embeddings)
similarity_scores = similarities[0].numpy()

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    # Where to put the dots (x, y, z coordinates)
    x=embeddings_3d[1:, 0],  # All rows except first (query), column 0 = x
    y=embeddings_3d[1:, 1],  # All rows except first, column 1 = y  
    z=embeddings_3d[1:, 2],  # All rows except first, column 2 = z
    
    # How to display them
    mode='markers+text',  
    
    # How the dots look
    marker=dict(
        size=10,                    
        color=similarity_scores,    # Color based on similarity (0.0-1.0)
        colorscale='Viridis',      # Color scheme (blue to yellow)
        showscale=True,            # Show the color bar legend
        colorbar=dict(title="Similarity")
    ),
    
    hovertext=documents,     # Full sentence appears on hover
    hoverinfo='text',        # Only show the hovertext (not x,y,z)
    
    # Short labels next to dots
    text=[f"Seq {i}: {similarity_scores[i]:.3f}" for i in range(len(documents))],
    
    name='Documents'  # Legend name
))

# Plot query (center) in red
fig.add_trace(go.Scatter3d(
    x=[embeddings_3d[0, 0]],  # Query is at index 0, x coordinate
    y=[embeddings_3d[0, 1]],  # Query y coordinate
    z=[embeddings_3d[0, 2]],  # Query z coordinate
    
    mode='markers+text',
    
    marker=dict(
        size=15,              # Bigger than documents
        color='red',          # Red color
        symbol='diamond',     # Diamond shape
        line=dict(color='black', width=2)  # Black outline
    ),
    
    text=['Query'],           # Label
    hovertext=[query],        # Full query text on hover!
    hoverinfo='text',
    
    name='Query (Center)'
))

fig.update_layout( # Remove legend for clarity
    title="3D Document Similarity Visualization",
    scene=dict(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        zaxis_title="Dimension 3"
    ),
    width=1000,
    height=800,
    showlegend=False 
)

fig.show()