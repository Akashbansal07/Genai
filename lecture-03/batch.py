from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
    )


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {area} expert, explain clear and concise"),
    ("human", " Explain {topic} in 5 sentence")
])

parser = StrOutputParser()

chain = prompt | llm | parser


topics=[
    {"area":"physics", "topic":"reflection"},
    {"area":"python", "topic":"dictionary"},
    {"area":"javascipt", "topic":"callback"},
    {"area":"maths", "topic":"acute angle"},
]


print("=======batch processing=====")

results = chain.batch(topics)

for i , result in enumerate(results):
    print(f"\n--------Topic {i+1} ------")
    print()
    print()
    print(result)
