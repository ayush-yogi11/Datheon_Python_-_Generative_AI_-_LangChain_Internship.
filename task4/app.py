import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables (.env should contain GEMINI_API_KEY)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Conversational AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Interactive AI Chatbot")
st.caption("Powered by Google Gemini & Streamlit with real-time streaming")

# 1. Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today?"}
    ]

# 2. Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if user_prompt := st.chat_input("Type your message here..."):
    # Render user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Prepare message history for Gemini (converting session state to LangChain format)
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))

    # 4. Stream Response from Gemini LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Initialize streaming Gemini client
            llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.7)
            
            # Stream tokens in real time
            for chunk in llm.stream(langchain_messages):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
            
            # Finalize message text without cursor
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"An error occurred: {str(e)}"
            message_placeholder.error(full_response)

    # Append AI response to session state history
    st.session_state.messages.append({"role": "assistant", "content": full_response})