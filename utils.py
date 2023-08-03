def load_document(file):
    import os

    print(f"Loading {file}...")
    _, extension = os.path.splitext(file)

    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(file)
    else:
        print(f"{extension} filetype not supported.")
        return None

    return loader.load()


def chunk_documents(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    chunks = RecursiveCharacterTextSplitter(
        chunk_overlap=chunk_overlap, chunk_size=chunk_size
    ).split_documents(data)

    return chunks


def create_embeddings(chunks):
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def ask(q, vector_store, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    return chain.run(q)


def calculate_embedding_cost(chunks):
    import tiktoken

    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    total_tokens = sum([len(encoding.encode(page.page_content)) for page in chunks])
    return total_tokens, total_tokens / 1000 * 0.0004


def clear_history():
    import streamlit as st

    if "history" in st.session_state:
        del st.session_state["history"]
