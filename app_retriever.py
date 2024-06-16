import os
import time
import json
import time
import base64
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
# Task
Engage in a natural conversation with the user based on their visit purpose. The query will include the visitor's name, the purpose of their visit, and their relationship with the AI. Use this information to craft your responses.

# Rules
1. Tailor your responses using appropriate expressions based on the visit purpose and the visitor's relationship with the AI.
2. Maintain the flow of the conversation by explaining naturally.
3. Never reveal that you are an AI assistant.
4. Verify the visitor's name, visit purpose, and relationship from the query, and continue the conversation accordingly.
5. Always respond in Japanese, even if you cannot find the correct answer.
6. Avoid repeating the same words or phrases.
7. Mention user's name only once in the first of the chatting.

# Examples
Query: 訪問者の名前は田中さん、目的は製品のデモを見ること、関係は既存の顧客です。
Response: 田中さん、こんにちは！製品のデモについてお聞きになりたいのですね。どの製品に興味がありますか？

Query: 訪問者の名前は鈴木さん、目的はサポート依頼、関係は新規顧客です。
Response: 鈴木さん、はじめまして。どのようなサポートが必要ですか？

Follow these rules and examples when answering user queries.

"""

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True,
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
    avatar_url = profile_data['avatar_url']

    hash_chat_name = base64.b64encode(chat_name.encode('utf-8'))
    hash_chat_name = hash_chat_name.decode('ascii')

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
            namespace_name = user_id + "-" + hash_chat_name
            #index_name = hashing.hash_value(origin_index_name, salt='secret')

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
                namespace=namespace_name ,
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
    query = "#実際情報" + "\n" + "あなたと話している相手の名前は" + user_name + "\n" + "相手があなたを訪問した目的、会話の話題は" + visit_purpose + "\n" +"あなたと訪問者の関係は" + relationship + "\n\n" + "#クエリ" + "\n" + request.json['query']

    chat_name = request.json['chat_name']

    hash_chat_name = base64.b64encode(chat_name.encode('utf-8'))
    hash_chat_name = hash_chat_name.decode('ascii')

    namespace = user_id + "-" + hash_chat_name

    vectorStore = PineconeVectorStore(index_name=user_id, embedding=embeddings, namespace=namespace)
    print(namespace, query)

    # retriever = vectorStore.as_retriever()
    
    # print(memory.buffer)
    # chain = ConversationalRetrievalChain.from_llm(llm, retriever= retriever, memory= conversational_memory)

    # res = chain.run({'question': query})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorStore.as_retriever()
    )

    tools = [
        Tool(
            name='AI Personal Info Retriever',
            func=qa.run,
            description=(
                "Use this tool to retrieve detailed information about the AI, including its background, capabilities, and other predefined details."
                "This tool should be used specifically for questions about the AI itself."
                "If the AI's name cannot be found, generate a response using the LLM but never reveal that you are an AI assistant."
                "For all other queries, respond based on the user's relationship and visit purpose."
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
    
    print("memory", memory)
    
    full_query = system_message + "\n\n" + query
    print("full_query--------------------->", full_query)

    answer = agent(full_query)
    return answer['output']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)