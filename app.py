import os
import time
import json
import pinecone
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

app = Flask(__name__)
CORS(app)

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
openai_api_key = os.environ.get('OPENAI_API_KEY')
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

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

pc = Pinecone(api_key=pinecone_api_key)
if use_serverless:  
    spec = ServerlessSpec(cloud='aws', region='us-east-1')  
else:  
    # if not using a starter index, you should specify a pod_type too  
    spec = PodSpec() 
# check for and delete index if already exists  
index_name = 'langchain-retrieval-augmentation-fast'  
if index_name in pc.list_indexes().names():  
    pc.delete_index(index_name)  
# create a new index  
pc.create_index(  
    index_name,  
    dimension=1536,  # dimensionality of text-embedding-ada-002  
    metric='dotproduct',  
    spec=spec  
)  
# wait for index to be initialized  
while not pc.describe_index(index_name).status['ready']:  
    time.sleep(1) 

@app.route('/create', methods=['POST'])
def create_chat():
    index = pc.Index(index_name)
    print(index.describe_index_stats())

    profile_data = request.json

    extra_keys = ['entry_name', 'doc_id', 'selected_model', 'avatar_url', 'model_name']

    with open('setting.json', 'r', encoding='utf8') as f:
        data = json.load(f)

    train_data = []

    for profile in profile_data:
        if profile not in  extra_keys:
            train_data.append({
                "id": str(len(train_data)),
                "question": data[profile],
                "answer": profile_data[profile]
            })

    for train_data_unit in train_data:
        medadatas = [{
            'data_id': train_data_unit['id'],
            'question': train_data_unit['question'],
            'answer': train_data_unit['answer'],
        }]
        documents = [json.dumps(train_data_unit)]
        embeds = embeddings.embed_documents(documents)
        ids = [train_data_unit['id']]

        index.upsert(vectors=zip(ids, embeds, medadatas))

    print(index.describe_index_stats())
    return 'FAQs indexed successfully!'

@app.route('/chat', methods=['POST'])
def conversion_agent():
    query = request.json['query']
    text_field = "question"

    index = pc.Index(index_name)
    print(query, index.describe_index_stats())

    vectorStore = PineconeVectorStore(index, embeddings, text_field)

    result = vectorStore.similarity_search("what is your name")

    print(result)
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
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
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

    answer = agent(query)
    print(answer)
    return answer['output']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
