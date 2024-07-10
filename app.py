import os
import time
import json
import time
import base64
from pinecone import Pinecone, ServerlessSpec, PodSpec  
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

from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

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
    # Tasks
    Your task is to chat with a user based on the purpose of their visit.
    The visitor's name, purpose of visit, and relationship to you are included in the query and can be retrieved from the query.
    You must answer according to the purpose of the visit and their relationship with you.
    If a visitor asks a question about you, you must answer it using the visitor's information in the query.

    # Rules
    1. be correct based on the purpose of the user's visit and their relationship with you.
    2. explain yourself naturally, continuing the previous conversation.
    3. do not say that you are an AI assistant.
    4. the query should allow you to see the visitor's name, purpose of visit, and relationship between the visitor and the member, and you should verify this information and converse accordingly
    5. always answer in Japanese, even if you cannot find the correct answer.
"""

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True,
    ai_prefix=system_message
)

pc = Pinecone(api_key=pinecone_api_key)

if use_serverless:  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  
else:  
    # if not using a starter index, you should specify a pod_type too  
    spec = PodSpec() 

def create_pinecone_index(index_name):
    try:
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
                if profile not in  extra_keys:
                    texts = texts + str(setting_data[profile])+" "+str(profile_data[profile])+" "

            index_name = user_id
            namespace_name = user_id + "-" + hash_chat_name
            
            if index_name not in pc.list_indexes().names():
                create_pinecone_index(index_name=index_name)
                
            texts = [texts]
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
    user_id = request.json['user_id']
    user_name = request.json['user_name']
    relationship = request.json['relationship']
    visit_purpose = request.json['visit_purpose']
    question = request.json['query']
    query = question + ". 私は " +user_name +  ". ボットとの関係:"+relationship+". 訪問目的:"+visit_purpose +". "+ ". 日本語で答えなければならない。"

    chat_name = request.json['chat_name']

    hash_chat_name = base64.b64encode(chat_name.encode('utf-8'))
    hash_chat_name = hash_chat_name.decode('ascii')

    namespace = user_id + "-" + hash_chat_name

    starter = "This is my name. So if user asks my name, please find my name in pinecone vector store and provide it. something like My name is XXX. I am happy to help you today. The bot assists users with  the context of the context.\n"
    template = """Only reply to non-technical questions such as about bot name, birthday and information. If the question is about yourself like age or gender and etc, but you don't have any info, please reply with I have nothing but the name about myself. How can I help you with others?. For the technical question, find the anaswer only based on the input context.  If there is no relevant info in the doucment, please say 'Sorry, I can't help with that.'
    response with Japanese
    """
    end = """ Context: {context}
    Chat history: {chat_history}
    Human: {human_input}
    Your Response as Chatbot: """

    template += starter + end

    prompt = PromptTemplate(
        input_variables=["chat_hitory", "human_input", "context"],
        template=template
    )

    docSearch = PineconeVectorStore.from_existing_index(
        index_name=user_id,
        embedding=embeddings,
        namespace=namespace
    )
    docs = ""
    if namespace != "-1":
        condition = {"collection_name": namespace}

        docs = docSearch.similarity_search(query, k = 8)
        print(docs)
    
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    stuff_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt, memory=memory)
    output = stuff_chain({"input_documents": docs, "human_input": query}, return_only_outputs=False)

    return output["output_text"]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
