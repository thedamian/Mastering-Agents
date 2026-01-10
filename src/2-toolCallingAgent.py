import streamlit as st
from openai import OpenAI
import os
import json

# Configure OpenAI API with key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the weather function tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The temperature in {city} is 72°F and raining!"

# Define the function declaration for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# Streamlit UI
st.title("🤖 OpenAI Chatbot with Tool Calling")
st.caption("Ask me about the weather in any city!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Send message to OpenAI
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=st.session_state.messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    
    # Handle function calls
    if response_message.tool_calls:
        # Add assistant's response to messages
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response_message.tool_calls
            ]
        })
        
        # Execute each function call
        for tool_call in response_message.tool_calls:
            if tool_call.function.name == "get_weather":
                args = json.loads(tool_call.function.arguments)
                city = args["city"]
                function_response = get_weather(city)
                
                # Add function response to messages
                st.session_state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_response
                })
        
        # Get final response from model
        final_response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=st.session_state.messages,
            tools=tools
        )
        assistant_message = final_response.choices[0].message.content
    else:
        assistant_message = response_message.content
    
    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message)