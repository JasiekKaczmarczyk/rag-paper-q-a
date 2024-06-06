import streamlit as st
import glob
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import faiss
from rag_utils import create_vector_db, ask_question_rag, ask_question_baseline

@st.cache_resource
def database(paths: list[str], database_save_path="faiss_index"):
    if not os.path.exists(database_save_path):
        create_vector_db(paths, chunk_size=400, vector_db_save_path=database_save_path)

    # vector_db = faiss.FAISS.read_index(database_save_path)
    vector_db = faiss.FAISS.load_local(
        database_save_path, 
        embeddings=OllamaEmbeddings(model="nomic-embed-text", show_progress=True), 
        allow_dangerous_deserialization=True,
    )

    return vector_db

def main():
    st.set_page_config(layout="wide")
    st.title("Rag paper QA")

    paths = glob.glob("papers/*.pdf")
    vector_db = database(paths)

    question = st.text_input(label="Question")

    if st.button(label="Ask"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### RAG solution")
            answer = ask_question_rag(vector_db, question, model_name="llama3")
            st.write(answer)
        with col2:
            st.write("#### Baseline solution")
            answer = ask_question_baseline(question, model_name="llama3")
            st.write(answer)

if __name__ == "__main__":
    main()