from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


conversation_history=[
    SystemMessage(content="You are a tourist guide. keep answer concise")
]

user_input_1="tell me few place to visit in delhi"
conversation_history.append(HumanMessage(content=user_input_1))

response_1=llm.invoke(conversation_history)
conversation_history.append(AIMessage(content=response_1.content))

print(f"user: {user_input_1}")
print(f"GPT: {response_1.content[:300]}...")
print()

user_input_2 ="we have time to visit one place onyl, so suugest me one from those."
conversation_history.append(HumanMessage(content=user_input_2))

response_2=llm.invoke(conversation_history)
conversation_history.append(AIMessage(content=response_2.content))


print(f"user: {user_input_2}")
print(f"GPT: {response_2.content[:100]}...")
print()