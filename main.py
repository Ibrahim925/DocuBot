import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import utils
import os


def main():
    st.subheader("DocuBot 🤖")
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        uploaded_file = st.file_uploader("Upload a File", type=["pdf", "docx"])
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=2048,
            value=512,
            on_change=utils.clear_history,
        )
        k = st.number_input(
            "K", min_value=1, max_value=20, value=3, on_change=utils.clear_history
        )
        add_data = st.button("Embed Data", on_click=utils.clear_history)

        if uploaded_file and add_data:
            with st.spinner("Reading, chunking, and embedding file..."):
                bytes_data = uploaded_file.read()
                file_name = os.path.join("./", "data", uploaded_file.name)
                with open(file_name, "wb") as f:
                    f.write(bytes_data)

                data = utils.load_document(file_name)
                chunks = utils.chunk_documents(data, chunk_size=chunk_size)
                st.write(f"Chunk size: {chunk_size}, chunks: {len(chunks)}")
                _, embedding_cost = utils.calculate_embedding_cost(chunks)
                st.write(f"Embedding cost: {embedding_cost:.4f}")

                vector_store = utils.create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success("File embedding was successful!")

    question = st.text_input("Ask a question")

    if question:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            answer = utils.ask(question, vector_store, k)
            st.text_area(f"LLM Answer: {answer}")

            st.divider()

            if "history" not in st.session_state:
                st.session_state.history = ""

            value = f"Q: {question}\nA: {answer}"
            st.session_state.history = (
                f"{value} \n {'-' * 100} \n {st.session_state.history}"
            )
            h = st.session_state.history
            st.text_area(label="Chat History", value=h, key="history", height=400)


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)
    main()
