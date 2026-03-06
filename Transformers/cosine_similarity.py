from sentence_transformers import SentenceTransformer,util
from sklearn.metrics.pairwise import cosine_similarity

sentences = ["I love AI", "AI is amazing", "I dislike bugs"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
similarity_matrix = cosine_similarity(embeddings)

similaity_score_1 = util.cos_sim(embeddings[0], embeddings[1]).item()
similaity_score_2 = util.cos_sim(embeddings[0], embeddings[2]).item()
similaity_score_3 = util.cos_sim(embeddings[1], embeddings[2]).item()
print(f"similarity score between {sentences[0]} and {sentences[1]}: {similaity_score_1:.4f}")
print(f"similarity score between {sentences[0]} and {sentences[2]}: {similaity_score_2:.4f}")
print(f"similarity score between {sentences[1]} and {sentences[2]}: {similaity_score_3:.4f}")