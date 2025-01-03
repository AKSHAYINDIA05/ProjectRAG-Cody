from unstructured.partition.pdf import partition_pdf
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.embeddings import JinaEmbeddings
from pymongo import MongoClient
import uuid
from langchain.schema.document import Document

class Chunking:
    def __init__(self, file:str):
        self.file = file
        self.elements = []
        self._images = None
        self._tables = None
        self._texts = None
        self._get_elements()

    def _get_elements(self):
        try:
            self.elements = partition_pdf(
                filename=self.file,
                infer_table_structure=True,
                strategy='hi_res',
                extract_image_block_types=['Image'],
                chunking_strategy='by_title',
                extract_image_block_to_payload=True
            )
        except Exception as e:
            raise RuntimeError(f'Error Getting the elements of the PDF : {e}')
    
    def _extract_texts_and_tables(self):
        if self._texts is None or self._tables is None:
            self._texts = []
            self._tables = []
            for chunk in self.elements:
                if "Table" in str(type(chunk)):
                    self._tables.append(chunk)
                else:
                    self._texts.append(chunk)

    def _extract_images(self):
        if self._images is None:
            self._images = []
            for chunk in self.elements:
                if "CompositeElement" in str(type(chunk)):
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Image" in str(type(el)):
                            self._images.append(el.metadata.orig_elements)
    
    def get_texts(self):
        self._extract_texts_and_tables()
        return self._texts
    
    def get_tables(self):
        self._extract_texts_and_tables()
        return self._tables
    
    def get_images(self):
        self._extract_images()
        return self._images
    
class Summarize:
    def __init__(self, texts, tables, images):
        self.texts = texts
        self.tables = tables
        self.images = images
        self.prompt_text = """
        You are an assistant tasked with summarizing tables and text.
        Give a concise summary of the table or text. Make sure you are not missing out any information.

        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.

        Table or text chunk: {element}

        """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_text)

        # Summary chain
        self.model = ChatGoogleGenerativeAI(
            api_key="AIzaSyAYT3i602ZoMBST61Ub7yh_RZF_pt2-ij8",
            model="models/gemini-1.5-flash-8b"
        )
        
        self.summarize_chain = {"element": lambda x: x} | self.prompt | self.model | StrOutputParser()

    def summarize_texts(self):
        try:
            return self.summarize_chain.batch(self.texts, {"max_concurrency": 3})
        except Exception as e:
            print(f"Error summarizing texts: {e}")
            return []
    
    def summarize_tables(self):
        try:
            self.tables_html = [table.metadata.text_as_html for table in self.tables]
            return self.summarize_chain.batch(self.tables_html, {"max_concurrency": 3})
        except Exception as e:
            print(f"Error summarizing tables: {e}")
            return []

    def summarize_images(self):
        try:
            self.image_prompt_template = """Describe the image in detail. For context,
                    the image is part of a research paper explaining the transformers
                    architecture. Be specific about graphs, such as bar plots."""
            self.messages = [
                (
                    "user",
                    [
                        {"type": "text", "text": self.image_prompt_template},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image}"},
                        },
                    ],
                )
            ]
            self.image_prompt = ChatPromptTemplate.from_messages(self.messages)
            chain = self.image_prompt | self.model | StrOutputParser()
            return chain.batch(self.images)
        
        except Exception as e:
            print(f"Error summarizing images: {e}")
            return []

class StoreInVectorDB:
    def __init__(self, key, texts, text_summaries, tables, table_summaries, images,  image_summaries):
        self.key = key
        self.texts = texts
        self.text_summaries = text_summaries
        self.tables = tables
        self.table_summaries = table_summaries
        self.images = images
        self.image_summaries = image_summaries

        self._initialize_db()
        self._initialize_vectorstore()

        self.collection.delete_many({})

        self._store_data("text", self.texts, self.text_summaries)
        self._store_data("table", self.tables, self.table_summaries)
        self._store_data("image", self.images, self.image_summaries)

    def _initialize_db(self):
        try:
            self.client = MongoClient(
                host="mongodb+srv://akshaymww:ADn8AFFTmQuH6QJg@clustermachinelearning.rlgyr.mongodb.net/?retryWrites=true&w=majority&appName=ClusterMachineLearning"
                )
            self.db = self.client['RAG_DB']
            self.collection = self.db[f'{self.key}_collection']
        except Exception as e:
            raise RuntimeError(f"Failed to connect to MongoDB: {e}")

    def _initialize_vectorstore(self):
        try:
            self.embedding_model = JinaEmbeddings(
                model_name='jina-embeddings-v3',
                jina_api_key="jina_094ac2007e584a649a4511d3ef54781fHEyKLdpU_69r8vKBVdj8NPUL2dvH"
            )
        
            self.vectorstore = MongoDBAtlasVectorSearch(
                collection=self.collection,
                embedding=self.embedding_model
            )

            self.store = InMemoryStore()
            self.id_key = "doc_id"

            self.retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.store,
                id_key=self.id_key
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store: {e}")
        
    def _store_data(self, datatype, data, summaries):
        try:
            ids = [str(uuid.uuid4()) for _ in data]
            documents = [Document(
                page_content=summary, metadata={self.id_key:ids[i], "type":datatype}) for i, summary in enumerate(summaries)
            ]
            self.retriever.vectorstore.add_documents(documents)
            self.retriever.docstore.mset(list(zip(ids, data)))
        except Exception as e:
            print(f"Error storing {datatype} data: {e}")
