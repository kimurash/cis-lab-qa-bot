import os
import time
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests

from splitter import PageContentSplitter

load_dotenv('../.env')

# Cosense API
base_url = 'https://scrapbox.io'
# Cosense のプロジェクト名
project_name = os.getenv('COSENSE_PROJECT_NAME')
# Cosense APIを叩くためのCookie
connect_sid = os.getenv('COSENSE_CONNECT_SID')

def create_db():
    # ページのタイトル一覧を取得する
    page_title_list = get_page_title_list()

    # 検索対象となるドキュメント
    docs = []
    for idx, page_title in enumerate(page_title_list):
        # ぺージの内容を取得する
        docs += create_documents(page_title)

        # 無料プランでは15[requests/minute]
        if idx % 14 == 0:
            time.sleep(60)
    
    # ドキュメントをデータベースに保存する
    store_db('faiss_store', docs)

def get_page_title_list() -> list[str]:
    response = (
        requests.get(
            f"{base_url}/api/pages/{project_name}/search/titles",
            cookies={
                'connect.sid': connect_sid
            }
        )
    )
    if response.status_code == 200:
        page_list = response.json()
        return [ page['title'] for page in page_list ]
    else:
        return []

def create_documents(page_title: str) -> List[Document]:
    page_content = get_page_content(page_title)

    if page_content is None:
        return []
    
    # Singletonパターンで実装しているため
    # 複数のインスタンスが生成されることはない
    content_splitter = (
        PageContentSplitter(
            headers_to_split_on=[
                ("#",   "Header 1"),
                ("##",  "Header 2"),
                ("###", "Header 3")
            ]
        )
    )
    # ページの内容を分割する
    docs = content_splitter.split_text(page_content)
    # ページの内容に話題のパンくずリストを追加する
    # 話題1 > 話題2 : ページの内容
    docs = [ add_topic_path(doc) for doc in docs ]

    return docs

def get_page_content(page_title: str):
    response = (
        requests.get(
            f"{base_url}/api/pages/{project_name}/{page_title}/text",
            cookies={
                'connect.sid': os.getenv('COSENSE_CONNECT_SID')
            }
        )
    )
    if response.status_code == 200:
        return response.text
    else:
        return None

def add_topic_path(doc: Document):
    breadcrumbs = ' > '.join(list(doc.metadata.values()))
    doc.page_content = breadcrumbs + " : " + doc.page_content
    return doc

def store_db(folder_path: str, docs: List[Document]):
    embeddings = (
        GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            model="models/embedding-001"
        )
    )
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(folder_path)

def load_db():
    embeddings = (
        GoogleGenerativeAIEmbeddings(
            google_api_key=os.getenv('GOOGLE_API_KEY'),
            model="models/embedding-001"
        )
    )
    return FAISS.load_local('faiss_store', embeddings, allow_dangerous_deserialization=True)
