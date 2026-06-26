import streamlit as st
from rag_pipeline import load_vectorstore, build_chain, check_hallucination

st.set_page_config(page_title="NLP Tutor Chatbot", page_icon="🎓")
st.title("🎓 NLP Domain Tutor Chatbot")
st.caption("Powered by RAG + Groq — Ask anything from the NLP course materials")

with st.sidebar:
    st.header("⚙️ Settings")
    # Use secret if deployed on Streamlit Cloud, else ask user
    groq_api_key = st.secrets.get("GROQ_API_KEY", None) or st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if not st.secrets.get("GROQ_API_KEY", None):
        st.markdown("[Get free Groq API key](https://console.groq.com)")
    st.divider()
    st.markdown("**About**")
    st.markdown("Answers questions grounded in NLP course syllabus and textbook chapters with hallucination detection.")

@st.cache_resource(show_spinner="Loading course materials...")
def get_vectorstore():
    return load_vectorstore("./syllabus_docs")

vectorstore, embeddings = get_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            color = "🔴" if meta["is_hallucination"] else "🟢"
            st.caption(f"{color} Faithfulness: {meta['faithfulness_score']} | Sources: {', '.join(set(meta['sources']))}")

if prompt := st.chat_input("Ask a question about the NLP course..."):
    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain, retriever = build_chain(vectorstore, groq_api_key)
            answer = chain.invoke(prompt)
            source_docs = retriever.invoke(prompt)

            hallucination = check_hallucination(answer, source_docs, embeddings)
            sources = [d.metadata.get("source", "unknown") for d in source_docs]

            st.markdown(answer)
            color = "🔴" if hallucination["is_hallucination"] else "🟢"
            st.caption(f"{color} Faithfulness: {hallucination['faithfulness_score']} | Sources: {', '.join(set(sources))}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {**hallucination, "sources": sources}
    })
