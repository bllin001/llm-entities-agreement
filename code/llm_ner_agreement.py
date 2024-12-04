import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import matplotlib.patches as patches

# Load the JSON data
with open('data/extracted_entities.json', 'r') as file:
    data = json.load(file)

# Flatten the entities into a dataframe for processing
entities = []
for model_data in data:
    model_name = model_data["model"]
    for entity_info in model_data["response"]:
        entities.append({
            "model": model_name,
            "entity": entity_info["entity"],
            "type": entity_info["type"]
        })

df = pd.DataFrame(entities)

# Initialize the transformer-based similarity model
from transformers import pipeline
similarity_model = pipeline('feature-extraction', model='sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for each entity
entity_embeddings = {}
for entity in df["entity"]:
    embedding = similarity_model(entity)[0]  # Get token embeddings
    entity_embeddings[entity] = np.mean(embedding, axis=0)  # Take the mean of token embeddings

# Compute cosine similarity matrix
entities_list = [f"{row['model']} - {row['entity']}" for _, row in df.iterrows()]
num_entities = len(entities_list)
similarity_matrix = np.zeros((num_entities, num_entities))

for i in range(num_entities):
    for j in range(num_entities):
        vec1 = entity_embeddings[df["entity"].iloc[i]]
        vec2 = entity_embeddings[df["entity"].iloc[j]]
        # Compute cosine similarity
        similarity_matrix[i, j] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Create a DataFrame for the similarity matrix
df_similarity = pd.DataFrame(
    similarity_matrix,
    index=entities_list,
    columns=entities_list
)

# Threshold-based clustering
threshold = 0.7
clusters = []
used_indices = set()

for i in range(num_entities):
    if i in used_indices:
        continue
    cluster = [i]  # Start a new cluster
    for j in range(num_entities):
        if i != j and similarity_matrix[i, j] >= threshold:
            cluster.append(j)
            used_indices.add(j)
    clusters.append(cluster)
    used_indices.add(i)

# Reorder the matrix based on clusters
clustered_indices = [idx for cluster in clusters for idx in cluster]
reordered_similarity = df_similarity.values[clustered_indices, :][:, clustered_indices]
reordered_labels = [entities_list[i] for i in clustered_indices]

# Plot the heatmap with clusters and add black boxes
plt.figure(figsize=(14, 12))
ax = sns.heatmap(
    reordered_similarity,
    cmap="vlag",
    xticklabels=reordered_labels,
    yticklabels=reordered_labels,
    annot=True,
    fmt=".2f",
    cbar_kws={"label": "Cosine Similarity"},
    # linewidths=0.5,
    # linecolor='black'
)

# Add black rectangles for clusters
for cluster in clusters:
    start_idx = clustered_indices.index(cluster[0])
    end_idx = clustered_indices.index(cluster[-1])
    rect = patches.Rectangle(
        (start_idx, start_idx),  # Bottom-left corner
        end_idx - start_idx + 1,  # Width
        end_idx - start_idx + 1,  # Height
        fill=False,
        edgecolor='black',
        linewidth=3
    )
    ax.add_patch(rect)

# Finalize the plot
plt.title("Cosine Similarity Heatmap of Entities (LLM - Entity)", fontsize=16, pad=20)
plt.xlabel("", fontsize=12)
plt.ylabel("", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("image/cosine_similarity_heatmap_clustered_with_threshold.png", dpi=300)
plt.show()

# Generate unique entities by grouping based on similarity
unique_entities = []
used_indices = set()

for i in range(num_entities):
    if i in used_indices:
        continue
    current_entity = entities_list[i]
    group = [current_entity]
    for j in range(num_entities):
        if i != j and j not in used_indices:
            if similarity_matrix[i, j] > threshold:
                group.append(entities_list[j])
                used_indices.add(j)
    # Add the most representative entity (longest name) from the group
    representative_entity = max(group, key=len)
    unique_entities.append({
        "entity": representative_entity.split(" - ")[1],
        "model": representative_entity.split(" - ")[0],
        "type": df[df["entity"] == representative_entity.split(" - ")[1]]["type"].iloc[0]
    })
    used_indices.add(i)

# Convert unique entities into a DataFrame
unique_entities_df = pd.DataFrame(unique_entities)

# Display the unique entities
print("Unique Entities:")
print(unique_entities_df)

# Save unique entities to a JSON file
unique_entities_df.to_json("data/unique_entities.json", orient="records", indent=4)