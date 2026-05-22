import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedder = OpenAIEmbeddings(model="text-embedding-3-large")




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