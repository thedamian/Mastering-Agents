import streamlit as st
import google.generativeai as genai

# Configure Gemini API with key from secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Define the weather function tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The temperature in {city} is 72°F."

# Define the function declaration for Gemini
weather_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather for a city",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "city": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The name of the city"
                    )
                },
                required=["city"]
            )
        )
    ]
)

# Initialize the model with tools
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    tools=[weather_tool]
)

# Streamlit UI
st.title("🤖 Gemini Chatbot with Tool Calling")
st.caption("Ask me about the weather in any city!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat = model.start_chat(history=[])

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
    
    # Send message to Gemini
    response = st.session_state.chat.send_message(prompt)
    
    # Handle function calls
    while response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        
        # Execute the function
        if function_call.name == "get_weather":
            city = function_call.args["city"]
            function_response = get_weather(city)
            
            # Send function response back to model
            response = st.session_state.chat.send_message(
                genai.protos.Content(
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name="get_weather",
                            response={"result": function_response}
                        )
                    )]
                )
            )
    
    # Display assistant response
    assistant_message = response.text
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    with st.chat_message("assistant"):
        st.markdown(assistant_message)