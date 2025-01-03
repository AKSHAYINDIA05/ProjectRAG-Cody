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

if file1 is not None and file2 is not None:
    with open("temp1.pdf", 'wb') as f:
        f.write(file1.read())
    with open('temp2.pdf', 'wb') as f:
        f.write(file2.read())
    query = st.text_input("Chat with me for Contextual Comparison.")
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

        print(file2_results)

        comparison_prompt = PromptTemplate(
        template="""
                Perform a contextual comparison between the two documents provided based on the user query.
                Start introducing yourself first. Your name is Cody, and you are here to perform 
                contextual comparison. Your sole purpose is to compare the contexts of the 
                two documents provided to you and output the exact differences between the two contexts.

                The user query is: {query}

                Document 1 context:
                {document1_context}

                Document 2 context:
                {document2_context}

                **Instructions for Response:**
                1. Your responses must highlight the differences between the two contexts exactly where they occur.
                2. For each difference, state the relevant content from both Document 1 and Document 2, side by side, and use clear markers (like bold or colors) to indicate differences.
                3. Use Markdown to format your response:
                    - Use `**bold text**` to highlight the exact differences.
                    - If the differences can be quantified or categorized, use a table format.

                **Strict Instructions:**
                - Only compare the given contexts. If the query is irrelevant to the contexts, reject the comparison politely.
                - Clearly state: "Sorry, I can only compare the given document contexts. Please ask a question related to them."

                Example Response Format:
                ### Key Differences
                | **Aspect**       | **Document 1**                                        | **Document 2**                                        |
                |------------------|-------------------------------------------------------|-------------------------------------------------------|
                | Concept Focus    | **AI: Focuses on creating human-like intelligence.**  | **ML: Focuses on learning from data.**               |
                | Applications     | **AI: Applied in diagnostics and fraud detection.**   | **ML: Used for disease prediction and retail.**      |
                | Challenges       | **AI: Faces ethical concerns.**                       | **ML: Struggles with overfitting.**                  |

                ### Differences Highlighted:
                - In Document 1: "**AI focuses on creating machines that replicate human intelligence**."
                - In Document 2: "**ML emphasizes algorithms to learn and improve from data**."

                Based on the differences, provide meaningful insights in the comparison, but do not generate unrelated content or answer queries not relevant to the documents.
            """,
            input_variables=["document1_context", "document2_context", "query"]
        )

        comparison_chain = (
            comparison_prompt | model | StrOutputParser()
        )

        results = comparison_chain.invoke({"document1_context":file1_results, "document2_context":file2_results, "query":query})

        tab1, tab2, tab3 = st.tabs(['Results', f'{file1.name} Chunks', f'{file2.name} Chunks'])

        with tab1:
            st.markdown(results)
        with tab2:
            st.markdown(file1_results)
        with tab3:
            st.markdown(file2_results)