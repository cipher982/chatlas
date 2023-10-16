import os

import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from streaming import StreamHandler

from chatlas import utils
from chatlas.agent.chatlas_sql import create_chatlas
from chatlas.data_prep import records, semantic


st.set_page_config(page_title="Chatlas", page_icon="🌎")
st.header("Chat over your location history")
st.write("This app uses langchain and AI to answer questions on your location history.")
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/cipher982/chatlas)"  # noqa
)


class StreamlitApp:
    def __init__(self):
        utils.configure_openai_api_key()
        # self.openai_model = "gpt-3.5-turbo"
        self.openai_model = "gpt-4"

    def save_file(self, file):
        folder = "tmp"
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = f"./{folder}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner("Processing your data...")
    def process_data(self):
        # Check if processed semantic places data has been generated
        if not semantic.DEFAULT_PLACES_OUTPUT_PATH.exists():
            placeholder = st.empty()
            placeholder.text("Generating processed data for semantic places...")
            semantic.main()
            placeholder.empty()

        # Check if processed semantic activities data has been generated
        if not semantic.DEFAULT_ACTIVITIES_OUTPUT_PATH.exists():
            placeholder = st.empty()
            placeholder.text("Generating processed data for semantic activities...")
            semantic.main(load_sql=True)
            placeholder.empty()

        # Check if processed granular records data has been generated
        if not records.DEFAULT_OUTPUT_PATH.exists():
            placeholder = st.empty()
            placeholder.text("Generating processed data for records...")
            records.main()
            placeholder.empty()

    @st.spinner("Connecting to AI...")
    def setup_agent(self):
        # Process data
        self.process_data()

        db_path = f"sqlite:///{semantic.SQL_DB_PATH}"
        model = "gpt-3.5-turbo-0613"
        # model = "gpt-4"
        llm = ChatOpenAI(client=None, model=model, temperature=0, streaming=True)
        agent = create_chatlas(llm=llm, db=db_path, functions=True)
        return agent

    @utils.enable_chat_history
    def main(self):
        if "agent" not in st.session_state:
            st.session_state.agent = self.setup_agent()

        user_query = st.chat_input(placeholder="Ask me about your travel history!")

        if user_query:
            utils.display_msg(user_query, "user")

            with st.chat_message("assistant"):
                response = st.session_state.agent.invoke({"input": user_query})
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
                st.experimental_rerun()


if __name__ == "__main__":
    obj = StreamlitApp()
    obj.main()
