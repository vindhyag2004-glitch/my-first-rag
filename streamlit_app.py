import streamlit as st
import numpy as np
import google.generativeai as genai
from pypdf import PdfReader

st.set_page_config(page_title="Multi-Doc Intelligence RAG", layout="wide")
st.title("📚 Conversational Multi-Doc Intelligence Assistant")

# 1. Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.info("👈 Enter your Gemini API key in the left sidebar to start. Get one at https://aistudio.google.com/app/apikey")
    st.stop()

genai.configure(api_key=api_key)

top_k = st.sidebar.slider("Number of context chunks to retrieve (Top-K):", min_value=1, max_value=5, value=3)

# 2. Embedding Model Resolution
@st.cache_resource
def get_embedding_model_name():
    try:
        for m in genai.list_models():
            if "embedContent" in m.supported_generation_methods:
                return m.name
    except Exception:
        pass
    return "models/embedding-001"

embed_model = get_embedding_model_name()

def get_doc_embedding(text):
    try:
        res = genai.embed_content(
            model=embed_model,
            content=text[:2000],
            task_type="retrieval_document"
        )
    except Exception:
        res = genai.embed_content(
            model=embed_model,
            content=text[:2000]
        )
    return res["embedding"]

def get_query_embedding(text):
    try:
        res = genai.embed_content(
            model=embed_model,
            content=text[:2000],
            task_type="retrieval_query"
        )
    except Exception:
        res = genai.embed_content(
            model=embed_model,
            content=text[:2000]
        )
    return res["embedding"]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_generative_model():
    for name in ["gemini-3.8-flash", "models/gemini-3.8-flash", "gemini-3.5-flash", "models/gemini-3.5-flash"]:
        try:
            return genai.GenerativeModel(name)
        except Exception:
            continue
    return genai.GenerativeModel("models/gemini-3.8-flash")

# 3. Text Chunking with Overlap
def chunk_text(text, chunk_size=700, chunk_overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

# 4. Sidebar - Document Ingestion
st.sidebar.header("📁 Document Ingestion")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs", 
    type=["pdf"], 
    accept_multiple_files=True
)

file_fingerprint = [f.name + str(f.size) for f in uploaded_files] if uploaded_files else []

if uploaded_files:
    if "file_fingerprint" not in st.session_state or st.session_state["file_fingerprint"] != file_fingerprint:
        all_chunks = []
        status_text = st.sidebar.empty()
        status_text.text("Extracting and windowing text...")

        for file in uploaded_files:
            reader = PdfReader(file)
            for page_idx, page in enumerate(reader.pages):
                txt = page.extract_text()
                if not txt:
                    continue
                page_chunks = chunk_text(txt)
                for c in page_chunks:
                    all_chunks.append({
                        "text": c,
                        "source": file.name,
                        "page": page_idx + 1
                    })

        if all_chunks:
            status_text.text("Generating vector embeddings...")
            progress_bar = st.sidebar.progress(0.0)
            embeddings = []

            for i, chunk in enumerate(all_chunks):
                embeddings.append(get_doc_embedding(chunk["text"]))
                progress_bar.progress((i + 1) / len(all_chunks))

            progress_bar.empty()
            st.session_state["chunks_data"] = all_chunks
            st.session_state["embeddings"] = embeddings
            st.session_state["file_fingerprint"] = file_fingerprint
            status_text.empty()
            st.sidebar.success(f"Indexed {len(all_chunks)} sections across {len(uploaded_files)} PDF(s)!")
        else:
            st.sidebar.warning("No readable text found in the uploaded document(s).")

# Clear chat memory option
if st.sidebar.button("🧹 Clear Conversation History"):
    st.session_state["messages"] = []
    st.rerun()

# 5. Chat Interface
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("🔍 View Cited Sources"):
                for src in message["sources"]:
                    st.markdown(f"**📄 Document:** `{src['source']}` | **Page:** `{src['page']}`")
                    st.caption(f"> {src['text']}")
                    st.divider()

user_query = st.chat_input("Ask a question or follow-up across your documents...")

if user_query:
    if not uploaded_files or "chunks_data" not in st.session_state:
        st.warning("Please upload at least one PDF first.")
    else:
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            model = get_generative_model()

            # Query Condensation: Reformulate query if chat history exists
            search_query = user_query
            if len(st.session_state["messages"]) > 1:
                with st.spinner("Refining search query from chat context..."):
                    history_str = "\n".join(
                        [f"{m['role'].upper()}: {m['content']}" for m in st.session_state["messages"][-5:-1]]
                    )
                    condense_prompt = (
                        "Given the chat history and a follow-up question, rewrite the follow-up into a standalone search query. "
                        "Do NOT answer the question. Only return the standalone query string.\n\n"
                        f"Chat History:\n{history_str}\n\n"
                        f"Follow-up Question: {user_query}\n\n"
                        "Standalone Search Query:"
                    )
                    try:
                        condense_res = model.generate_content(condense_prompt)
                        search_query = condense_res.text.strip()
                    except Exception:
                        search_query = user_query

            with st.spinner("Searching document context and generating answer..."):
                query_emb = get_query_embedding(search_query)
                scores = [cosine_similarity(query_emb, emb) for emb in st.session_state["embeddings"]]
                
                top_indices = np.argsort(scores)[::-1][:top_k]
                retrieved_chunks = [st.session_state["chunks_data"][i] for i in top_indices]

                context_parts = [
                    f"[Source {i}: {c['source']}, Page {c['page']}]\n{c['text']}"
                    for i, c in enumerate(retrieved_chunks, 1)
                ]
                retrieved_context = "\n\n---\n\n".join(context_parts)

                # Recent conversational turns for final prompt
                recent_history = "\n".join(
                    [f"{m['role'].upper()}: {m['content']}" for m in st.session_state["messages"][-4:]]
                )

                prompt = (
                    "You are a precise, conversational document intelligence assistant. "
                    "Answer the user query thoroughly and accurately using the context excerpts below and relevant prior discussion. "
                    "Cite the document name and page number for facts you reference. "
                    "If the answer cannot be determined from the context, state clearly that it is not in the documents.\n\n"
                    f"Context Excerpts:\n{retrieved_context}\n\n"
                    f"Conversation So Far:\n{recent_history}\n\n"
                    f"Assistant Answer:"
                )

                try:
                    response = model.generate_content(prompt)
                    answer = response.text
                except Exception as e:
                    answer = f"Error generating answer: {e}"

                st.markdown(answer)
                with st.expander("🔍 View Cited Sources"):
                    st.caption(f"**Search Query Used:** `{search_query}`")
                    st.divider()
                    for src in retrieved_chunks:
                        st.markdown(f"**📄 Document:** `{src['source']}` | **Page:** `{src['page']}`")
                        st.caption(f"> {src['text']}")
                        st.divider()

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_chunks
                })