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



def run_chatbot():
    print("\n" + "="*50)
    print("Travel advisor chatbot")
    print("Type 'quit' to exit | type 'history' to see all message")
    print("="*50 + "\n")

    session_config= {"configurable": {"session_id": "live_session"}}

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "history":
            history = store.get("live_session")
            if history:
                print(f"\n Conversation history----------")
                for msg in history.messages:
                    role = "You" if msg.type == "human" else "GPT"
                    print(f"{role}:{msg.content[:80]}...")
                print()
            continue


        print("GPT:", end="",flush=True)
        for chunk in chain_with_memory.stream(
            {"user_input": user_input},
            config=session_config
        ):
            print(chunk, end="", flush=True)
        print()
        print()

if __name__ == "__main__":
    run_chatbot()