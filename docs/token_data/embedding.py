from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import euclidean
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
similarities = model.similarity(query_embeddings, document_embeddings)

pairs = [] # Used later for sorting and printing 

print(f"Query embedding shape: {query_embeddings.shape} (1 query × 768 dimensions)")
print(f"Document embeddings shape: {document_embeddings.shape} ({len(documents)} documents × 768 dimensions)")

print("-----\nDocuments and their similarity scores (Highest to lowest):")
for i in range(len(documents)):
    pairs.append((float(similarities[0][i]), documents[i])) # [0] is the query and [i] is the document 
    # Goes through and adds each similarity and document as a tuple to the pairs list, iterating by document 
pairs.sort(reverse=True) # Sort by highest similarity 

for similarity, document in pairs:
    print(f"{round(similarity, 3)}  -  {document}") # Print each similarity and document 

print("-----\nEuclidean Distances:")
for i, document in enumerate(documents): # Loop through documents
    dist = euclidean(query_embeddings, document_embeddings[i]) # Calculate euclidean distance
    print(f"Doc {i}: {dist:.3f} - {document[:60]}...") # Print the euclidean distance and then print first 60 characters of document

print("-----\nTop 3 most similar documents:")
converted_array = similarities[0].numpy()
# Turns the array from a tensor to a numpy array so np can work with it
top3 = np.argsort(converted_array)[-3:][::-1] # [-3:] gets last 3 (highest), [::-1] reverses order
for rank, idx in enumerate(top3, 1): 
    # Example: top3 = [0, 13, 7]
    # The 1 let's the top 3 start at rank 1 instead of 0
    print(f"{rank}. Doc {idx}")
    print(f"   Cosine Similarity: {float(similarities[0][idx]):.3f}")
    print(f"   Euclidean Distance: {euclidean(query_embeddings, document_embeddings[idx]):.3f}")
    print(f"   {documents[idx]}\n")