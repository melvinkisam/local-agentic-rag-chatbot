# import basics
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
import streamlit as st

load_dotenv()

# initialize the Ollama embeddings model
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)

# initialize the Chroma vector store
vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)

# initialize chat model
llm = init_chat_model(
    os.getenv("CHAT_MODEL"),
    model_provider=os.getenv("MODEL_PROVIDER"),
    temperature=0
)

# prompt template for the agent
prompt = """                                
You are a helpful assistant. You will be provided with a query and a chat history.
Your task is to retrieve relevant information from the vector store and provide a response.
For this you use the tool 'retrieve' to get the relevant information.
                                      
The query is as follows:                    
{input}

The chat history is as follows:
{chat_history}

Please provide a concise and informative response based on the retrieved information.
If you don't know the answer, say "I don't know" (and don't provide a source).
                                      
You can use the scratchpad to store any intermediate results or notes.
The scratchpad is as follows:
{agent_scratchpad}

For every piece of information you provide, also provide the source.

Return text as follows:

<Answer to the question>
Source: filename.txt
"""


# creating the retriever tool
@tool
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)

    serialized = ""

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        serialized += f"Source: {source}\nContent: {doc.page_content}\n\n"

    return serialized

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_agent(model=llm, tools=tools, system_prompt=prompt)

# initiating streamlit app
st.set_page_config(page_title="Local Agentic RAG Chatbot", page_icon="🤖")
st.title("🤖 Local Agentic RAG Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
user_question = st.chat_input("How can I help you today?")

# did the user submit a prompt?
# if user_question:

#     # add the message from the user (prompt) to the screen with streamlit
#     with st.chat_message("user"):
#         st.markdown(user_question)

#         st.session_state.messages.append(HumanMessage(user_question))

#     # invoking the agent
#     result = agent.invoke({
#         "messages": [
#             {"role": "user", "content": user_question}
#         ]
#     })

#     ai_message = result['messages'][-1].content

#     # adding the response from the llm to the screen (and chat)
#     with st.chat_message("assistant"):
#         st.markdown(ai_message)

#         st.session_state.messages.append(AIMessage(ai_message))



if user_question:

    # add user message to screen
    with st.chat_message("user"):
        st.markdown(user_question)

    # convert previous messages + append current input
    chat_history = [
        {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else
        {"role": "assistant", "content": msg.content} 
        for msg in st.session_state.messages
    ]
    chat_history.append({"role": "user", "content": user_question})

    # invoke agent with full chat history
    result = agent.invoke({"messages": chat_history})

    # get last AIMessage
    ai_message = next(
        (msg.content for msg in reversed(result['messages']) if isinstance(msg, AIMessage)),
        "No response from agent."
    )

    # update chat history
    st.session_state.messages.append(HumanMessage(user_question))
    st.session_state.messages.append(AIMessage(ai_message))

    # display assistant message
    with st.chat_message("assistant"):
        st.markdown(ai_message)
