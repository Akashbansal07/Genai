from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv
from datetime import date
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o" , temperature=0)

class ProductReview(BaseModel):
    product_name: str          = Field(description="Name of the product being reviewed")
    rating:       int          = Field(description="Rating from 1 to 5, inferred from sentiment")
    pros:         List[str]    = Field(description="List of positive points mentioned")
    cons:         List[str]    = Field(description="List of negative points mentioned")
    sentiment:    str          = Field(description="overall, positive, negative, or mixed")
    recommend:    bool         = Field(description="True if the reviewer recommends the product")


review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a product review analyser.
Extract structured information from customer reviews.
Be accurate — only include information that is actually present in the review.
For rating, infer a 1-5 score from the overall sentiment if not explicitly stated."""),
    ("human", "Analyse this review:\n\n{review_text}")
])

review_structured_llm = llm.with_structured_output(ProductReview)

review_chain= review_prompt | review_structured_llm


reviews = [
    "I bought the Sony WH-1000XM5 headphones last month. The noise cancellation is absolutely incredible — I can work in a coffee shop like I am in a silent room. Battery life is great, easily 30 hours. The only downside is the ear cups get a bit warm after 2 hours. Worth every rupee!",
    "This laptop is a disaster. Overheats after 20 minutes, the keyboard is mushy, and it crashes randomly. Save your money."
]


for review in reviews:
    result= review_chain.invoke({"review_text": review})
    print(f"Rating: {result.rating}/5")
    print(f"\nProduct: {result.product_name}")
