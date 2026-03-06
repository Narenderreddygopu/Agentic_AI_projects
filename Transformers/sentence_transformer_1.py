from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
print(embeddings.shape)
print(f"Embedding for first sentence: {embeddings[0]}")
print(f"Embedding for second sentence: {embeddings[1]}")
print(f"Embedding dimension: {embeddings.shape[1]}")
print
(f"Embedding for first sentence: {embeddings[0][:10]}...")  # Print first 10 dimensions