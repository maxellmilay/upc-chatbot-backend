from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Hello world! My name is John Smith and I live in New York City. I have 25 apples."

embedding = model.encode(text)

print(embedding)
