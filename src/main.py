import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

from db import create_db
from db import load_db


def main():
    load_dotenv('../.env')

    # データベースがなければ作成
    if not os.path.exists('faiss_store'):
        create_db()

    db = load_db()
    qa = init_qa(db.as_retriever())

    init_page()

    # sessionにこれまでのメッセージを格納する

    # ユーザーの入力を監視
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 変数に値を代入してその値を使って条件を評価する
    if user_input := st.chat_input('質問を入力して下さい'):
        # 以前のチャットログを表示
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # ユーザーの入力を表示
        with st.chat_message('user'):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        # AIの応答を表示
        with st.chat_message('assistant'):
            with st.spinner('Gemini is typing ...'):
                response = qa.invoke(user_input)
            st.markdown(response['result'])

        st.session_state.messages.append({"role": "assistant", "content": response["result"]})

def init_page():
    st.set_page_config(
        page_title='Geminiを活用したRAGアプリケーション',
        page_icon="👑"
    )
    st.header('情報知能システム研究室QAボット')

def init_qa(retriever: VectorStoreRetriever):
    llm = (
        ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0,
            max_retries=2,
        )
    )
    qa = (
        RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
    )
    return qa

if __name__ == '__main__':
    main()
