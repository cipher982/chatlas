import os

import pandas as pd
import streamlit as st
import utils
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from streaming import StreamHandler

from chatlas.data_prep import records, semantic

st.set_page_config(page_title="Chatlas", page_icon="🌎")
st.header("Chat over your location history")
st.write("This app uses langchain and AI to answer questions on your location history.")
st.write(
    "[![view source code ](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/cipher982/chatlas)"
)


class ChatlasBot:
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"

    def save_file(self, file):
        folder = "tmp"
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = f"./{folder}/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner("Loading your data...")
    def setup_qa_chain(self):
        # Check if processed data has been generated
        if not semantic.DEFAULT_OUTPUT_PATH.exists():
            # Generate processed data
            semantic.main()

        # Check if processed data has been generated for records
        if not records.DEFAULT_OUTPUT_PATH.exists():
            # Generate processed data
            records.main()

        # Load processed data
        df = pd.read_pickle(semantic.DEFAULT_OUTPUT_PATH)

        # Create langchain agent
        llm = ChatOpenAI(model=self.openai_model, temperature=0, streaming=True)

        agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True)

        # # Setup memory for contextual conversation
        # memory = ConversationBufferMemory(
        #     memory_key='chat_history',
        #     return_messages=True
        # )

        # Setup LLM and QA chain
        # llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        # qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

        return agent

    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            qa_chain = self.setup_qa_chain()

            utils.display_msg(user_query, "user")

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    obj = ChatlasBot()
    obj.main()
