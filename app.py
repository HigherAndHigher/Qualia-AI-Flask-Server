import os
import time
import json
import time
import pinecone
from pinecone import Pinecone, ServerlessSpec, PodSpec  
#from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_pinecone import PineconeVectorStore 
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

app = Flask(__name__)
CORS(app)

#Configure FireStore
cred = credentials.Certificate('./key.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

#env variables
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')

#configure pinecone and langchain
use_serverless = True  

model_name = 'text-embedding-ada-002'  

embeddings = OpenAIEmbeddings(  
    model=model_name,  
    openai_api_key=openai_api_key  
)

llm = ChatOpenAI(
    openai_api_key = os.environ['OPENAI_API_KEY'],
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

system_message = """
    # タスク
    あなたの仕事は、ユーザーの訪問目的に基づいてユーザーとチャットすることです。
    訪問者の名前、訪問目的、あなたとの関係はクエリに含まれており、クエリから取得することができます。
    あなたは、訪問目的とあなたとの関係に応じて答える必要があります。

    # ルール
    1. ユーザーの訪問目的とあなたとの関係に基づいて、正しく表現すること。
    2. 前の会話を続けながら、自然に説明すること。
    3. 自分がAIアシスタントであるとは言わないこと。
    4. 回答は訪問者の名前から始めるべきである。
    5. 正しい答えが見つからなくても、常に日本語で答えなさい」
"""

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True,
    ai_prefix=system_message
)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True, len=5)

pc = Pinecone(api_key=pinecone_api_key)

if use_serverless:  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  
else:  
    # if not using a starter index, you should specify a pod_type too  
    spec = PodSpec() 

def create_pinecone_index(index_name):
    try:
        print(index_name, pc.list_indexes().names())
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
    
    try:
        info = db.collection('chat_models').where(filter=FieldFilter('user_id', '==', user_id)).where(filter=FieldFilter('chat_name', '==', chat_name))
        print(len(info.get()))
        if len(info.get()) > 0:
            print("Exist", info)
            return {
                "success": False
            }
        else:
            extra_keys = ['user_id', 'entry_name', 'doc_id', 'selected_model', 'avatar_url', 'model_name']

            with open('setting.json', 'r', encoding='utf8') as f:
                setting_data = json.load(f)

            texts = ""

            for profile in profile_data:
                if profile not in  extra_keys:
                    texts = texts + str(setting_data[profile])+" "+str(profile_data[profile])+" "

            texts = [texts]
            index_name = user_id
            namespace_name = user_id + "-" + chat_name
            #index_name = hashing.hash_value(origin_index_name, salt='secret')

            if index_name not in pc.list_indexes().names():
                create_pinecone_index(index_name=index_name)

            vectordb = PineconeVectorStore.from_texts(texts=texts, embedding=embeddings, index_name=index_name, namespace=namespace_name)
            db.collection('chat_models').add({
                "user_id": user_id,
                "chat_name": chat_name,
                "index_name": index_name,
                "namespace_name": namespace_name
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

@app.route('/chat', methods=['POST'])
def conversion_agent():
    print(request.json)
    user_id = request.json['user_id']
    user_name = request.json['user_name']
    relationship = request.json['relationship']
    visit_purpose = request.json['visit_purpose']
    query = request.json['query']
    #query = query + ". 私は " +user_name +  ". ボットとの関係:"+relationship+". 訪問目的:"+visit_purpose +". 訪問者の名前がクエリにある. 回答は訪問者の名前から始めるべきである。"+ ". 日本語で答えなければならない。"
    query += " いつも日本語で答える"

    chat_name = request.json['chat_name']
    namespace = user_id + "-" + chat_name

    vectorStore = PineconeVectorStore(index_name=user_id, embedding=embeddings, namespace=namespace)
    print(namespace, query)

    # retriever = vectorStore.as_retriever()
    
    # print(memory.buffer)
    # chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= conversational_memory)

    # res = chain.run({'question': query, })
    # print("result >>>>>>>", res)

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
                'use this tool when answering about your personal information such as name, birthday, birthplace, work experience, job and so on to get more information about the topic'
                ' visitor name is' + user_name +' '
                ' visit purpose of visior is' + visit_purpose + ' '
                ' assume relationship between you and visitor as' + relationship + ' '
                " answer about visitor's question"
                ' always answer with japanese even though you have not foound the correct answer' 
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
        memory=conversational_memory,
    )

    answer = agent(query)
    return answer['output']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
