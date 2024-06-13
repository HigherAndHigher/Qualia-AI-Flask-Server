import os
import time
import json
import base64
from pinecone import Pinecone, ServerlessSpec, PodSpec  
from langchain.vectorstores import Pinecone as PineconeVectorStore 
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferWindowMemory

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

app = Flask(__name__)
CORS(app)

# Configure FireStore
cred = credentials.Certificate('./key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Environment variables
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Configure Pinecone and LangChain
use_serverless = True  
model_name = 'text-embedding-ada-002'

embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=openai_api_key  
)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

system_message = """
# タスク
ユーザーの訪問目的に基づいて、ユーザーと自然なチャットを行ってください。
クエリには訪問者の名前、訪問目的、訪問者との関係が含まれています。それを基に回答してください。

# ルール
1. 訪問目的と訪問者との関係に基づいて、適切な表現を使って答えてください。
2. 会話の流れを保ちながら、自然に説明してください。
3. 自分がAIアシスタントであることを明かさないでください。
4. クエリから訪問者の名前、訪問目的、訪問者との関係を確認し、それに応じて会話を続けてください。
5. 正しい答えが見つからなくても、常に日本語で答えてください。

# 例
クエリ: 訪問者の名前は田中さん、目的は製品のデモを見ること、関係は既存の顧客です。
応答: 田中さん、こんにちは！製品のデモについてお聞きになりたいのですね。どの製品に興味がありますか？

クエリ: 訪問者の名前は鈴木さん、目的はサポート依頼、関係は新規顧客です。
応答: 鈴木さん、はじめまして。どのようなサポートが必要ですか？

これらのルールと例に従って、ユーザーの質問に答えてください。
"""

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True
)

pc = Pinecone(api_key=pinecone_api_key)

if use_serverless:  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  
else:  
    spec = PodSpec()

def create_pinecone_index(index_name):
    try:
        if index_name in pc.list_indexes().names():
            return False
        else:    
            pc.create_index(
                index_name,
                dimension=1536,
                metric='dotproduct',
                spec=spec
            )
            while not pc.describe_index(index_name).status['ready']:  
                time.sleep(5)
    except Exception as e:
        print(e)
        return False

    return True

@app.route('/create', methods=['POST'])
def create_chat():
    profile_data = request.json
    user_id = profile_data['user_id']
    chat_name = profile_data['model_name']
    avatar_url = profile_data['avatar_url']

    hash_chat_name = base64.b64encode(chat_name.encode('utf-8'))
    hash_chat_name = hash_chat_name.decode('ascii')

    try:
        info = db.collection('chat_models').where(filter=FieldFilter('user_id', '==', user_id)).where(filter=FieldFilter('chat_name', '==', chat_name))
        if len(info.get()) > 0:
            return {
                "success": False
            }
        else:
            extra_keys = ['user_id', 'entry_name', 'doc_id', 'selected_model', 'avatar_url', 'model_name']

            with open('setting.json', 'r', encoding='utf8') as f:
                setting_data = json.load(f)

            texts = ""

            for profile in profile_data:
                if profile not in extra_keys:
                    texts = texts + str(setting_data[profile]) + " " + str(profile_data[profile]) + " "

            texts = [texts]
            index_name = user_id
            namespace_name = user_id + "-" + hash_chat_name

            if index_name not in pc.list_indexes().names():
                create_pinecone_index(index_name=index_name)

            vectordb = PineconeVectorStore.from_texts(texts=texts, embedding=embeddings, index_name=index_name, namespace=namespace_name)
            db.collection('chat_models').add({
                "user_id": user_id,
                "chat_name": chat_name,
                "index_name": index_name,
                "namespace_name": namespace_name,
                "avatar_url": avatar_url
            })

    except Exception as e:
        print("error", e)
        return {
            "success": False
        }

    return {
        "success": True,
        "index_name": index_name,
        "namespace_name": namespace_name
    }

@app.route('/delete', methods=['POST'])
def delete_chat():
    data = request.json
    chat_name = data['model_name']
    user_id = data['user_id']

    hash_chat_name = base64.b64encode(chat_name.encode('utf-8'))
    hash_chat_name = hash_chat_name.decode('ascii')

    namespace_name = user_id + '-' + hash_chat_name

    try:
        if user_id in pc.list_indexes().names():
            index = pc.Index(user_id)
            index.delete(
                delete_all=True,
                namespace=namespace_name,
            )
            return "success"
        else:    
            return "failed"
    except Exception as e:
        print(e)
        return "failed"

@app.route('/chat', methods=['POST'])
def conversion_agent():
    print(request.json)
    user_id = request.json['user_id']
    user_name = request.json['user_name']
    relationship = request.json['relationship']
    visit_purpose = request.json['visit_purpose']
    query = request.json['query']
    chat_name = request.json['chat_name']

    hash_chat_name = base64.b64encode(chat_name.encode('utf-8'))
    hash_chat_name = hash_chat_name.decode('ascii')

    namespace = user_id + "-" + hash_chat_name

    vectorStore = PineconeVectorStore(index_name=user_id, embedding=embeddings, namespace=namespace)
    print(namespace, query)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorStore.as_retriever()
    )

    tools = [
        Tool(
            name='Knowledge Base',
            func=qa.run,
            description=(
                '名前、誕生日、出身地、職業、趣味、職業などの個人情報を回答する際に、このツールを使用すると、トピックに関する詳細な情報を得ることができます。'
                ' あなたと話している相手の名前は' + user_name + ' '
                ' 相手があなたを訪問した目的、会話の話題は' + visit_purpose + ' '
                ' あなたと訪問者の関係を' + relationship + ' '
                " 質問に対する回答"
                ' 正しい答えが見つからなくても、常に日本語で答えること。'
            )
        )
    ]

    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )

    # Include system message in the query context
    full_query = system_message + "\n\n" + query

    answer = agent(full_query)
    return jsonify(answer['output'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
