from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv



load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


zero_shot_prompt= ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analyser. Classify the sentiment of the text."),
    ("human", "Text:{text}")
])


zero_shot_chain = zero_shot_prompt | llm | parser

texts = [
    "I am very excited to meet my friends",
    "I am really angry about what happened",
    "I feel so lonely"

]

for text in texts:
    result = zero_shot_chain.invoke({"text": text})
    print(f"Input:{text}" )
    print(f"Output:{result}")
    print()


