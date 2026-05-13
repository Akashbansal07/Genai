from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explain things simply."),
    ("human", " Explain {topic} in 2 sentence")
])

formatted = prompt.format_messages(topic="reflection")




for msg in formatted:
    print(f"{msg.content}")