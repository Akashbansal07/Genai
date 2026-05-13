from langchain_openai import ChatOpenAI
from langchain_core.output_parser import StrOutputParser
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
    )

parser = StrOutputParser()

response= llm.invoke("what is capital of india")
print(response.content)