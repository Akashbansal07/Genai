from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv
from datetime import date
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

llm = ChatOpenAI(model="gpt-4o" , temperature=0)




class Invoice(BaseModel):
    vendor: str = Field(description="Name of the company or person who sent the invoice")
    amount: float = Field(description="Total amount due as a number, no currency symbols")
    date: str  = Field(description="Invoice date in YYYY-MM-DD format")
    items: List[str] = Field(description="List of items or service billed")
    currency: str = Field(description="Currency code, eg INR, USD, EUR")

structured_llm = llm.with_structured_output(Invoice)

raw_invoice="""
Hi 
the invoice from Snitch Fashions, dated 19th may 2026 for the following items:
- 3 x shirts at $2000 each
- 4 x jeans at $ 1500 each

total amount due: $12000
payment due with in a week

regards,
snitch team
"""

result = structured_llm.invoke(
    f"Extract the invoice information from this text:\n\n{raw_invoice}"
)

print(f"Vendor: {result.vendor}")
print(f"Amount:{result.amount}")
print(f"Date: {result.date}")
print(f"Currency: {result.currency}")
print(f"Items: ")
for item in result.items:
    print(f"-{item}")






    




