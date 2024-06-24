import streamlit as st

st.title("Rule-Based Chatbot :)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize state to track user selections
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None
if "selected_info" not in st.session_state:
    st.session_state.selected_info = None
if "end_of_chat" not in st.session_state:
    st.session_state.end_of_chat = False
if "live_support" not in st.session_state:
    st.session_state.live_support = False

# Display chat messages from history on app return
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Function to display initial country options
def display_country_options():
    st.write("Please select a country:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("America"):
            st.session_state.selected_country = "America"
            st.rerun()  # Immediately rerun the script after selection
    with col2:
        if st.button("Canada"):
            st.session_state.selected_country = "Canada"
            st.rerun()  # Immediately rerun the script after selection
    with col3:
        if st.button("Japan"):
            st.session_state.selected_country = "Japan"
            st.rerun()  # Immediately rerun the script after selection

# Function to display information options based on selected country
def display_info_options(country):
    st.write(f"You selected {country}. What information would you like?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Weather in {country}"):
            st.session_state.selected_info = f"Weather in {country}"
            st.rerun()  # Immediately rerun the script after selection
    with col2:
        if st.button(f"First Language in {country}"):
            st.session_state.selected_info = f"First Language in {country}"
            st.rerun()  # Immediately rerun the script after selection

# Function to get response based on user's selections
def get_response(country, info):
    if info.startswith("Weather"):
        if country == "America":
            return "The weather in America varies greatly depending on the region."
        elif country == "Canada":
            return "Canada has a diverse climate, with cold winters and warm summers."
        elif country == "Japan":
            return "Japan has four distinct seasons with a temperate climate."
    elif info.startswith("First Language"):
        if country == "America":
            return "The primary language in America is English."
        elif country == "Canada":
            return "The primary languages in Canada are English and French."
        elif country == "Japan":
            return "The primary language in Japan is Japanese."

# Handle the live support state
if st.session_state.live_support:
    with st.chat_message("assistant"):
        st.markdown("An agent will be in touch with you by email soon.")
    st.session_state.messages.append({"role": "assistant", "content": "An agent will be in touch with you by email soon."})
    st.stop()  # Stop further execution to prevent returning to the menu

# React to user input based on selections
if st.session_state.selected_country and st.session_state.selected_info:
    response = get_response(st.session_state.selected_country, st.session_state.selected_info)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.end_of_chat = True  # Indicate the end of chat
    st.session_state.selected_country = None
    st.session_state.selected_info = None
else:
    if not st.session_state.selected_country:
        display_country_options()
    else:
        display_info_options(st.session_state.selected_country)

# Display options at the end of the chat
if st.session_state.end_of_chat:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Return to the menu"):
            st.session_state.end_of_chat = False
            st.rerun()  # Rerun the script to reset the state and display the initial menu
    with col2:
        if st.button("Connect with live support agents"):
            st.session_state.live_support = True
            st.rerun()

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({'role': "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
