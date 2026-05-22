import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedder = OpenAIEmbeddings(model="text-embedding-3-large")

def cosine(vec_a: list, vec_b:list) -> float:
    a = np.array(vec_a)
    b= np.array(vec_b)

    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    return dot_product / (magnitude_a * magnitude_b)


sentences =[
    " love travelling ",
    "i will travel hyderbad next month",
    "Hyderabad is a tech city",
    "I like to code in python",
    "I want to go on world tour",

]

vectors = embedder.embed_documents(sentences)

anchor = sentences[0]
anchor_vec = vectors[0]


print(f"ancchor: {anchor}")
print()

print("-"*80)

for i in  range(1, len(sentences)):
    sim = cosine(anchor_vec, vectors[i])
    bar="█" * int(sim*20)
    print(f"{sentences[i]:<50} {sim:>8.3f} {bar}")