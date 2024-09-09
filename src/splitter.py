import os
from typing import List

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv('../.env')

class PageContentSplitter:
    # Geminiモデル
    llm = (
        ChatGoogleGenerativeAI(
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            model="gemini-1.5-flash",
            temperature=0.0,
            max_retries=2,
        )
    )
    # Geminiに渡すプロンプト
    messages = [
        (
            "system",
            "与えられた文書をMarkdown形式に変換してください。"
            "文書内の情報はすべてマークダウンに含めてください。"
            "ページのタイトルを見出し1とし、"
            "ページの内容に応じて適切に階層化された見出しを追加してください。"
            "リストや強調が必要な箇所は、適切なマークダウン記法を用いて視覚的にわかりやすく表現してください。"
        ),
        (
            "human",
            "{user_input}"
        ),
    ]

    # インスタンスを格納しておくクラス変数
    instance = None

    # Singletonパターン
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            # objectクラスを継承しているためsuper()はobjectを指す
            cls.instance = super().__new__(cls)

        return cls.instance

    def __init__(self, *args, **kwargs):
        # Markdownを見出しで分割するクラス
        self.md_splitter = MarkdownHeaderTextSplitter(*args, **kwargs)

    def split_text(self, text: str, messages: list[tuple]=None) -> List[Document]:
        if messages is None:
            messages = self.messages

        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | self.llm

        # ページの内容をMarkdown形式に変換する
        response = (
            chain.invoke(
                {
                    "user_input": text,
                }
            )
        )
        # Markdownを見出しで分割する
        docs = self.md_splitter.split_text(response.content)
        
        return docs
