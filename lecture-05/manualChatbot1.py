from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are a tourist guide. keep answer concise"),


    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{user_input}")
])

chain = prompt | llm | StrOutputParser()


store={}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns the message history for a given session.
    Creates a new empty history if the session does not exist yet,
    this function is called automatically by RunnableWithMessageHistory
    """

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id] 



chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)



session_1= {"configurable":{"session_id":"Tourist_a"}}


response= chain_with_memory.invoke(
    {"user_input": "Tell me place to visit in himachal pardesh"},
    config= session_1
)

print(f"user: Tell me place to visit in himachal pardesh ")
print(f"GPT: {response}")
print()