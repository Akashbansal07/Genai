# ── imports ───────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

# ── model ─────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── prompt template with message history placeholder ──────
# MessagesPlaceholder is special — it inserts the full
# conversation history into the prompt at that position
# "chat_history" is the variable name — must match
# what we pass to RunnableWithMessageHistory below
prompt = ChatPromptTemplate.from_messages([
    # system message: sets persona for the whole conversation
    ("system", "You are a helpful Python tutor. Keep your answers concise and clear."),

    # MessagesPlaceholder inserts all previous messages here
    # this is how the model sees the conversation history
    # it expands to: [HumanMessage, AIMessage, HumanMessage, AIMessage, ...]
    MessagesPlaceholder(variable_name="chat_history"),

    # current user message — the new question
    ("human", "{user_input}")
])

# ── chain ─────────────────────────────────────────────────
# standard LCEL chain: prompt | llm | parser
chain = prompt | llm | StrOutputParser()

# ── message history store ─────────────────────────────────
# this dict maps session_id → InMemoryChatMessageHistory
# each session_id is a separate conversation
# this is how you support multiple users simultaneously
# key = session_id (string), value = message history object
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns the message history for a given session.
    Creates a new empty history if the session does not exist yet.
    This function is called automatically by RunnableWithMessageHistory.
    """
    if session_id not in store:
        # first time we see this session — create empty history
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ── chain with memory ─────────────────────────────────────
# RunnableWithMessageHistory wraps our chain with automatic
# message history management:
# 1. before invoking: loads history for the session, injects into prompt
# 2. after invoking:  saves the new human + AI messages to history
# input_messages_key  = which key in our input dict is the user message
# history_messages_key = which MessagesPlaceholder to fill with history
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_input",
    history_messages_key="chat_history"
)


# ── live terminal chatbot ─────────────────────────────────
# this is a real interactive chatbot in the terminal
# type a message, get a response, type another message
# the model remembers everything you said in this session
# type 'quit' or 'exit' to stop

def run_chatbot():
    """
    Interactive terminal chatbot with full conversation memory.
    Every message is remembered for the duration of the session.
    """
    print("\n" + "="*50)
    print("  Python Tutor Chatbot")
    print("  Type 'quit' to exit | Type 'history' to see all messages")
    print("="*50 + "\n")

    # unique session id for this chatbot run
    # in a real app this would be the logged-in user's ID
    session_config = {"configurable": {"session_id": "live_session"}}

    while True:
        # get input from the user
        user_input = input("You: ").strip()

        # handle special commands
        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "history":
            # show the full conversation history so far
            history = store.get("live_session")
            if history:
                print(f"\n--- Conversation history ({len(history.messages)} messages) ---")
                for msg in history.messages:
                    role = "You" if msg.type == "human" else "GPT"
                    print(f"  {role}: {msg.content[:80]}...")
                print()
            continue

        # send the message and stream the response word by word
        print("GPT: ", end="", flush=True)
        for chunk in chain_with_memory.stream(
            {"user_input": user_input},
            config=session_config
        ):
            print(chunk, end="", flush=True)
        print()  # newline after response
        print()  # blank line for readability

if __name__ == "__main__":
    run_chatbot()