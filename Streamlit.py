    user_input = st.chat_input("Ask a question related to data, quering, data warehouse and analysis")

 

    if user_input:

        # Append user input to conversation history

        st.session_state.conversation_history.append(f"User: {user_input}")

 

        with st.chat_message("User"):

            st.markdown(user_input)
