import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain

import os

from personalized_web_research import PersonalizedWebResearch

os.environ["GOOGLE_API_KEY"] = "" # Get it at https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
os.environ["GOOGLE_CSE_ID"] = "" # Get it at https://programmablesearchengine.google.com/
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
os.environ["OPENAI_API_KEY"] = "" # Get it at https://beta.openai.com/account/api-keys

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

def settings():

    # Vectorstore
    import faiss
    from langchain.vectorstores import FAISS 
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore import InMemoryDocstore  
    embeddings_model = OpenAIEmbeddings()  
    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()

    # personalization


    # Initialize 
    web_retriever = PersonalizedWebResearch.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=3
    )



    return web_retriever, llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


st.sidebar.image("img/ai.png")
st.header("`Personalized Interweb Explorer`")
st.info("`I am an AI that can answer questions tailored to your personalized knowledge")

# Make retriever and llm
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

personalization_url = st.text_input("`Where are you comimg from? Share a link that is telling about you:`")

import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)

specialty = "general audience"

if personalization_url:
    from langchain.document_loaders import WebBaseLoader
    from langchain.indexes import VectorstoreIndexCreator
    loader = WebBaseLoader(personalization_url)
    index = VectorstoreIndexCreator().from_loaders([loader])
    sector = index.query("what is the industry sector for this person or organization")
    logging.info(f"Industry sector for personalization: {sector}")
    if not sector.endswith('know.'):
        start_phrase_pos = sector.find(" in the")
        if start_phrase_pos > 0:
            specialty = sector[start_phrase_pos+8:len(sector)-1]
        else:
            specialty = sector.split()[-1]

        specialty_result =  st.empty()
        specialty_result.info('`Sector:`\n\n' + sector + '\n\n`Extracted specialty:`\n\n' + specialty )

# User input 
question = st.text_input("`Ask a question:`")

if question:

    # Generate answer (w/ citations)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    question += '$$$ ' + specialty
    result = qa_chain({"question": question, "specialty": specialty},callbacks=[retrieval_streamer_cb, stream_handler])
    answer.info('`Answer:`\n\n' + result['answer'])
    st.info('`Sources:`\n\n' + result['sources'])
