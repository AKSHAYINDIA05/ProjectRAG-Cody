from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import streamlit as st
from openai import OpenAI
from chunking import Chunking, Summarize, StoreInVectorDB
from pymongo import MongoClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatGoogleGenerativeAI(
            api_key="AIzaSyAYT3i602ZoMBST61Ub7yh_RZF_pt2-ij8",
            model="models/gemini-1.5-flash-8b"
        )

st.title("_:blue[cod]:green[Easy]_")
st.caption("By *_:blue[cod]:green[Easy]_*")
st.sidebar.title("_:blue[cod]:green[Easy]_")
file = st.sidebar.file_uploader("Upload your File", type=['pdf'])

if file is not None:
    with open("file.pdf", 'wb') as f:
        f.write(file.read())
    
    query = st.text_input("Chat with you File.", key="query")
    submit_query = st.button("Submit")

    if submit_query:
        temp_chunking = Chunking("file.pdf")
        temp_texts = temp_chunking.get_texts()
        temp_tables = temp_chunking.get_tables()
        temp_images = temp_chunking.get_images()

        summarizer = Summarize(texts=temp_texts,
                            tables=temp_tables,
                            images=temp_images)
        
        temp_text_summaries = summarizer.summarize_texts()
        temp_table_summaries = summarizer.summarize_tables()
        temp_image_summaries = summarizer.summarize_images()

        store = StoreInVectorDB(
            key="file",
            texts=temp_texts,
            text_summaries=temp_text_summaries,
            tables=temp_tables,
            table_summaries=temp_table_summaries,
            images=temp_images,
            image_summaries=temp_image_summaries
        )
    
    client = MongoClient(
                    host="mongodb+srv://akshaymww:ADn8AFFTmQuH6QJg@clustermachinelearning.rlgyr.mongodb.net/?retryWrites=true&w=majority&appName=ClusterMachineLearning"
                    )
    db = client['RAG_DB']
    file_collection = db[f'file_collection']
    
    embedding_url = "https://api.jina.ai/v1/embeddings"

    def generate_embedding(text:str) -> List[float]:
        response = requests.post(
            embedding_url,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer jina_094ac2007e584a649a4511d3ef54781fHEyKLdpU_69r8vKBVdj8NPUL2dvH'
            },
            json={
                "input":text,
                "model":"jina-embeddings-v3"
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to generate embedding. Status code: {response.status_code}:{response.text}")
        
        return response.json()['data'][0]['embedding']
    
    file_results = file_collection.aggregate(
        [
        {
        "$vectorSearch":{
            "queryVector":generate_embedding(query),
            "path":"embedding",
            "numCandidates":10,
            "limit":3,
            "index":"fileSearchIndex"
            }
        }
        ]
    )

    file_texts = [result['text'] for result in file_results]

    file_results = "\n".join(file_texts)

    prompt = PromptTemplate(
    template=
        """
        You are an expert Chatbot designed for answering questions based on the provided context only.
        Answer the question based only on the following context, which can include text, tables, and the below image.
        The context provided to you is:
        `Context`: {context_text}
        The user question is:
        `Question`: {user_question}
        
        NOTE:Now you need to answer the question only from the provided context.
        In case of questions out of the context, you need to politely decline answering the user's question.
        
        """
    )

    comparison_chain = (
        prompt | model | StrOutputParser()
    )

    results1 = comparison_chain.invoke({"context_text":file_results, "user_question":query})

    tab1, tab2 = st.tabs(["Result", f'{file.name} Chunks'])
    with tab1:
        st.write(results1)
    with tab2:
        st.write(file_results)

if file is None:
    client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="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"
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "system",
            "content": "Your name is WinGPT. You are a chatbot created by WinWire."
        })
    for message in st.session_state.messages:
        if message['role'] != "system":
            with st.chat_message(message['role']):
                st.markdown(message['content'])
    if prompt := st.chat_input("Chat with me."):
        st.session_state.messages.append({"role":"user", "content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
