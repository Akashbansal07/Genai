from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv



load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


cot_shot_prompt= ChatPromptTemplate.from_messages([
    ("system", """You are a maths assistant. 
     Think throuh every problem step by step.
     show your working clearly.
     Answer with final number."""),
    ("human", "{problem}")
])


cot_shot_chain = cot_shot_prompt | llm | parser

problem="""
A sell apple $5 each and orange for $10 each.
Hussain buys 4 apples and 3 oranges.
He pays with a $100 note.
How much change does he get?
"""
result = cot_shot_chain.invoke({"problem": problem})
print(f"Answer:{result}")
print()

