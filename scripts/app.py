import streamlit as st
import requests
import json
import time

# Page Config
st.set_page_config(
    page_title="AetherAI",
    page_icon="🌌",
    layout="centered"
)

# Styling
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.title("Project Aether 🌌")
st.caption("Advanced Multimodal Intelligence • v2.5.0")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Message Aether..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # API Call
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Thinking..."):
                # Connect to local FastAPI server
                # Sending "User: ... \nSystem:" format to match training
                formatted_prompt = f"User: {prompt}\nSystem:"
                
                response = requests.post(
                    "http://localhost:8000/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": formatted_prompt}],
                        "max_tokens": 50,
                        "temperature": 0.7
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["choices"][0]["message"]["content"]
                    
                    # Clean up response (in case server didn't)
                    if "System:" in answer:
                        answer = answer.split("System:")[-1]
                    
                    # Streaming effect
                    for char in answer:
                        full_response += char
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
                    message_placeholder.markdown(full_response)
                else:
                    st.error(f"API Error: {response.status_code}")
                    full_response = "Error connecting to neural engine."
                    
        except requests.exceptions.ConnectionError:
            st.error("Server is offline. Run `python scripts/server.py` first.")
            full_response = "Connection refused."

        st.session_state.messages.append({"role": "assistant", "content": full_response})
