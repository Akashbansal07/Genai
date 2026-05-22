import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embedder = OpenAIEmbeddings(model="text-embedding-3-large")


faq_chunks = [
    "To cancel your subscription, go to Settings and click Cancel Plan.",
    "You can terminate your membership at any time from your account page.",
    "Refunds are processed within 5-7 business days after cancellation.",
    "Our premium plan includes unlimited API calls and priority support.",
    "To reset your password, click Forgot Password on the login page.",
    "Contact our support team at support@company.com for billing issues.",
    "Stop your recurring payment by visiting the billing section.",
    "New users get a 30-day free trial with no credit card required.",
]


faq_vectors = embedder.embed_documents(faq_chunks)


def cosine(vec_a: list, vec_b:list) -> float:
    a = np.array(vec_a)
    b= np.array(vec_b)

    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    return dot_product / (magnitude_a * magnitude_b)


def semantic_search (query: str, chunks: list,chunk_vectors: list, top_k: int=3) -> list:
    query_vector = embedder.embed_query(query)

    similarities=[]

    for i , chunk_vec in enumerate(chunk_vectors):
        score= cosine(query_vector, chunk_vec)
        similarities.append((chunks[i], score))

    
    similarities.sort(key=lambda x:x[1], reverse=True)

    return similarities[:top_k]


queries = [
    "How can i stop my subscription",
    "when will i get my money back",
    "I forgot my login details, what should i do"
]

for query in queries:
    print(f"Query: {query}")
    results = semantic_search(query, faq_chunks, faq_vectors, top_k=2)
    for chunk, score in results:
        print(f" score {score}:{chunk}")
    print()
