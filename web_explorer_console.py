from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain

import os

from personalized_web_research import PersonalizedWebResearch

os.environ[
    "GOOGLE_API_KEY"] = ""  # Get it at https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
os.environ["GOOGLE_CSE_ID"] = ""  # Get it at https://programmablesearchengine.google.com/
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
os.environ[
    "OPENAI_API_KEY"] = ""  # Get it at https://beta.openai.com/account/api-keys

# Vectorstore
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore

embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


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
# LLM
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

# Search
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()

# Initialize
web_retriever = PersonalizedWebResearch.from_llm(
    vectorstore=vectorstore_public,
    llm=llm,
    search=search,
    num_search_results=3
)


# Make retriever and llm
""" 
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm
"""

personalization_url = "https://www.oracle.com/artificial-intelligence/generative-ai/"

specialty = "medical doctor"

from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

loader = WebBaseLoader(personalization_url)
index = VectorstoreIndexCreator().from_loaders([loader])
sector = index.query("what is the industry sector for this person or organization")
specialty = sector.split()[-1]


# User input 
question = "what is generative AI"

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

question += '$$$ '+ specialty
result = qa_chain({"question": question, "specialty": question})

