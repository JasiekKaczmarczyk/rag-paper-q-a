import pypdf
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, faiss
from langchain_core.documents import Document

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

def parse_pdf_to_pages(path: str):
    with open(path, "rb") as pdfFileObj:
        pdfReader = pypdf.PdfReader(pdfFileObj)
        pages = []

        for i, page in enumerate(pdfReader.pages):
            pages.append(Document(page_content=page.extract_text()))
    
    return pages

def create_vector_db(paths: list[str], chunk_size: int = 1500, vector_db_save_path="faiss_index"):
    documents = []
    for path in paths:
        document = parse_pdf_to_pages(path)
        documents += document
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)

    chunks = text_splitter.split_documents(documents)

    # vector_db = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    #     collection_name="local_rag"
    # )
    vector_db = faiss.FAISS.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
    )
    vector_db.save_local(vector_db_save_path)

    # return vector_db

def ask_question_rag(vector_db: Chroma, question: str, model_name: str = "llama3"):

    llm = ChatOllama(model=model_name)

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from vector database. 
        By generating multiple perspectives on the user question, your goal is to help the user overome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=query_prompt,
    )

    template = """Answer the question based primarily on the following context and your training. Be as specific and as informative as you can:
    {context}
    Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    answer = chain.invoke(question)

    return answer

def ask_question_baseline(question: str, model_name: str = "llama3"):
    llm = ChatOllama(model=model_name)

    template = """Answer the question based on the your training:
    Question: {question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(question)

    return answer