import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedder = OpenAIEmbeddings(model="text-embedding-3-large")

#small- 1536 dimensions(cost : 0.02$ million token)
#large - 3072 dimesions (cost: 10times of small)


sentence ="I love GenAI"

vector = embedder.embed_query(sentence)

print(f"vector: {vector[:10]}")
print(f"vector type: {type(vector)}")
print(f"vector {len(vector)}")
print(f"vector {min(vector) } to {max(vector)}")