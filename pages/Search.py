from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import streamlit as st
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
file1 = st.sidebar.file_uploader("Upload your **FIRST** File", type=['pdf'])
file2 = st.sidebar.file_uploader("Upload your **SECOND** File", type=['pdf'])

with st.sidebar:
    st.markdown("<b>:blue[Powered by Gemini].<b>:red[(*for Document Based*)]<br><b>:blue[Powered by Llama].<b>:red[(*for Simple Text Based*)]", unsafe_allow_html=True)

if file1 is not None and file2 is not None:
    with open("temp1.pdf", 'wb') as f:
        f.write(file1.read())
    with open('temp2.pdf', 'wb') as f:
        f.write(file2.read())
    query = st.text_input("Type here to search your documents.")
    submit_query = st.button("Submit")

    if submit_query:
        temp1_chunking = Chunking("temp1.pdf")
        temp1_texts = temp1_chunking.get_texts()
        temp1_tables = temp1_chunking.get_tables()
        temp1_images = temp1_chunking.get_images()

        summarizer = Summarize(texts=temp1_texts,
                            tables=temp1_tables,
                            images=temp1_images)
        
        temp1_text_summaries = summarizer.summarize_texts()
        temp1_table_summaries = summarizer.summarize_tables()
        temp1_image_summaries = summarizer.summarize_images()

        store = StoreInVectorDB(
            key="file1",
            texts=temp1_texts,
            text_summaries=temp1_text_summaries,
            tables=temp1_tables,
            table_summaries=temp1_table_summaries,
            images=temp1_images,
            image_summaries=temp1_image_summaries
        )

        temp2_chunking = Chunking('temp2.pdf')
        temp2_texts = temp2_chunking.get_texts()
        temp2_tables = temp2_chunking.get_tables()
        temp2_images = temp2_chunking.get_images()

        summarizer = Summarize(texts=temp2_texts,
                            tables=temp2_tables,
                            images=temp2_images)
        
        temp2_text_summaries = summarizer.summarize_texts()
        temp2_table_summaries = summarizer.summarize_tables()
        temp2_image_summaries = summarizer.summarize_images()

        store = StoreInVectorDB(
            key="file2",
            texts=temp2_texts,
            text_summaries=temp2_text_summaries,
            tables=temp2_tables,
            table_summaries=temp2_table_summaries,
            images=temp2_images,
            image_summaries=temp2_image_summaries
        )

        client = MongoClient(
                    host="mongodb+srv://akshaymww:ADn8AFFTmQuH6QJg@clustermachinelearning.rlgyr.mongodb.net/?retryWrites=true&w=majority&appName=ClusterMachineLearning"
                    )
        db = client['RAG_DB']
        file1_collection = db[f'file1_collection']
        file2_collection = db[f'file2_collection']

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
        
        file1_results = file1_collection.aggregate(
            [
            {
            "$vectorSearch":{
                "queryVector":generate_embedding(query),
                "path":"embedding",
                "numCandidates":file1_collection.count_documents(filter={}),
                "limit":3,
                "index":"File1SearchIndex"
                }
            }
            ]
        )

        file2_results = file2_collection.aggregate(
            [
            {
            "$vectorSearch":{
                "queryVector":generate_embedding(query),
                "path":"embedding",
                "numCandidates":file2_collection.count_documents(filter={}),
                "limit":3,
                "index":"File2SearchIndex"
                }
            }
            ]
        )

        file1_texts = [result['text'] for result in file1_results]
        file2_texts = [result['text'] for result in file2_results]

        file1_results = "\n".join(file1_texts)
        file2_results = "\n".join(file2_texts)
        search_prompt = PromptTemplate(
            template="""
                "You are an intelligent assistant equipped with two distinct Documents. 
                Here is the content of *Document 1*:
                -> {document1_content}
                Here is the content of *Document 2*:
                -> {document2_content}
                And, here is the user's *query*:
                -> {query}
                Your task is to answer user queries based on the specified context while adhering to the following rules:

                1. **Query Interpretation**:
                - If the user specifies a context (e.g., 'From Document 1' or 'From Document 2'), restrict your answer to that context only.
                - If no document is specified, use both contexts to provide the most relevant information.

                2. **Response Generation**:
                - **When a Specific Document is Specified**:
                    - Analyze the query and extract only the relevant information from the designated document.
                    - Do not include or infer information from the other document.
                - **When No Document is Specified**:
                    - Analyze both contexts and combine relevant information in a clear and concise manner.
                    - Highlight any discrepancies or differences between the documents, if applicable.

                3. **Handling Out-of-Document Queries**:
                - If the requested information is not found in the specified document, respond politely:  
                    "The requested information is not available in the specified document. Please refine your query or review the document."

                4. **Formatting the Response**:
                - Clearly indicate the document used in the response (e.g., 'Based on Document 1:' or 'Based on Document 2:').
                - Use bullet points or concise sentences for clarity.
                - Provide direct and factual answers without unnecessary commentary.

                5. **Examples**:
                - **Query**: "From Document 1, what is the revenue for January?"  
                    **Document 1**: "The revenue for January was $10,000."  
                    **Document 2**: "January revenue was reported as $12,000."  
                    **Response**:  
                    "Based on Document 1:  
                    - The revenue for January was $10,000."

                - **Query**: "From Document 2, what were the marketing strategies?"  
                    **Document 1**: "The strategies included social media campaigns and email outreach."  
                    **Document 2**: "Marketing efforts focused on influencer collaborations."  
                    **Response**:  
                    "Based on Document 2:  
                    - Marketing efforts focused on influencer collaborations."

                - **Query**: "What is the location of the company?"  
                    **Document 1**: "The company is headquartered in New York."  
                    **Document 2**: "No information about the location is provided."  
                    **Response**:  
                    "Based on both documents:  
                    - Document 1 states the company is headquartered in New York.  
                    - Document 2 does not provide any information about the company's location."
            """,
            input_variables=["document1_content", "document2_content", "query"]
        )

        comparison_chain = (
            search_prompt | model | StrOutputParser()
        )

        results = comparison_chain.invoke({"document1_content":file1_results, "document2_content":file2_results, "query":query})

        tab1, tab2, tab3 = st.tabs(['Results', f'{file1.name} Chunks', f'{file2.name} Chunks'])

        with tab1:
            st.markdown(results)
        with tab2:
            st.markdown(file1_results)
        with tab3:
            st.markdown(file2_results)
