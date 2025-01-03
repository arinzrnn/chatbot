from __future__ import annotations
import os
import datetime
import pandas as pd
import streamlit as st
import openai
import typing as t
import requests
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_recall import ContextRecall
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from pydantic import BaseModel, ValidationError
from rouge_score import rouge_scorer
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

import asyncio

# Load environment variables
_ = load_dotenv(find_dotenv())

# Set OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Determine LLM model name based on date
current_date = datetime.datetime.now().date()
llm_name = "gpt-4-turbo" if current_date < datetime.date(2023, 9, 2) else "gpt-4-turbo"

# Initialize ChatOpenAI
llm = ChatOpenAI(
    model_name=llm_name,
    temperature=0,
    openai_api_key=openai_api_key,
    max_tokens=2000,
    max_retries=2  # Set max retries if needed
)

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4-turbo"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Prompt template for GPT-based answers
template = """Berikan penjelasan terlebih dahulu mengenai {question} apabila output berupa link, berdasarkan konteks yang tersedia.

Konteks: {context}

Question: {question}

Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# Function to load database from Excel
def load_db(file, chain_type, k):
    combined_text = ""

    if file.name.endswith(".xlsx"):
        # Load the Excel file
        xls = pd.ExcelFile(file)
        sheets_data = {}

        # Load all sheets and combine text
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            sheets_data[sheet_name] = df.to_string()  # Convert each sheet to a string

        # Combine all sheet data into one document
        combined_text = "\n".join(sheets_data.values())

    elif file.name.endswith(".pdf"):
        # Load the PDF file
        pdf_reader = PdfReader(file)
        pdf_text = []
        for page in pdf_reader.pages:
            pdf_text.append(page.extract_text())
        combined_text = "\n".join(pdf_text)

    else:
        raise ValueError("Unsupported file format. Please upload an Excel (.xlsx) or PDF (.pdf) file.")

    # Split the text for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(combined_text)

    # Convert chunks to Document objects
    documents = [Document(page_content=chunk) for chunk in text_chunks]

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = DocArrayInMemorySearch.from_documents(documents, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Initialize the retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )

    return qa

def debug_inputs(reference, context, response, user_input):
    print("---- DEBUG START ----")
    try:
        print(f"Reference type: {type(reference)}, Value: {reference}")
    except Exception as e:
        print(f"Error with 'reference': {e}")

    try:
        if isinstance(context, list):
            print(f"Context type: {type(context)}, Length: {len(context)}, Sample: {context[:2]}")
        else:
            print(f"Context type: {type(context)}, Value: {context}")
    except Exception as e:
        print(f"Error with 'context': {e}")

    try:
        print(f"Response type: {type(response)}, Value: {response}")
    except Exception as e:
        print(f"Error with 'response': {e}")

    try:
        print(f"User Input type: {type(user_input)}, Value: {user_input}")
    except Exception as e:
        print(f"Error with 'user_input': {e}")
    print("---- DEBUG END ----")


def evaluate_response_with_ragas(reference, context, response, user_input):
    def to_plain_string(value):
        if hasattr(value, 'to_string'):
            return value.to_string()
        if isinstance(value, str):
            return value
        return str(value)

    # Convert inputs to plain strings
    reference = to_plain_string(reference)
    response = to_plain_string(response)
    user_input = to_plain_string(user_input)

    # Ensure context is a list
    if isinstance(context, str):
        context = [context]

    debug_inputs(reference, context, response, user_input)

    # Initialize metrics with LLM
    faithfulness = Faithfulness(llm=evaluator_llm)
    context_precision = ContextPrecision(llm=evaluator_llm)
    answer_relevance = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_recall = ContextRecall(llm=evaluator_llm)

    sample = SingleTurnSample(
        reference=reference,
        retrieved_contexts=context,
        response=response,
        user_input=user_input
    )

    async def calculate_scores():
        faithfulness_score = await faithfulness._single_turn_ascore(sample, callbacks=None)
        faithfulness_score = faithfulness_score if faithfulness_score is not None else 0  # Default ke 0 jika None
        context_precision_score = await context_precision._single_turn_ascore(sample, callbacks=None)
        answer_relevance_score = await answer_relevance._single_turn_ascore(sample, callbacks=None)
        context_recall_score = await context_recall._single_turn_ascore(sample, callbacks=None)

        return {
            "Faithfulness": faithfulness_score,
            "Context Precision": context_precision_score,
            "Answer Relevance": answer_relevance_score,
            "Context Recall": context_recall_score,
        }

    scores = asyncio.run(calculate_scores())

    st.session_state["evaluation_metrics"].append({
        "Question": user_input,
        "Response": response,
        "Reference": reference,
        "Context": context,
        "Faithfulness": scores["Faithfulness"],
        "Context Precision": scores["Context Precision"],
        "Answer Relevance": scores["Answer Relevance"],
        "Context Recall": scores["Context Recall"]
    })

# Step 1: Definisi alat bantu untuk retrieve
def hitungdosis_tool(input_text: str) -> str:
    """
    Fungsi untuk menghitung dosis obat menggunakan GPT dalam BAHASA INDONESIA.
    Args:
        input_text (str): Input dari pengguna terkait perhitungan dosis.
    Returns:
        str: Jawaban dari GPT dalam bahasa Indonesia.
    """
    query = f"""
    Anda adalah seorang dokter anak yang ahli dalam memberikan perhitungan dosis obat.
    Berikan jawaban, pemikiran, dan observasi dalam Bahasa Indonesia.
    Hitung dosis obat dalam BAHASA INDONESIA berdasarkan informasi berikut:

    {input_text}

    **Format Jawaban yang Diharapkan (Dalam Bahasa Indonesia):**
    - **Sirup**: Nama sediaan, jumlah ml yang dibutuhkan, frekuensi (setiap berapa jam).      
    - **Drops**: Nama sediaan, jumlah ml dan tetes yang dibutuhkan, serta frekuensinya.
    - **Puyer**: Jumlah total tablet yang dibutuhkan untuk membuat 10 pertamen.
    - **Rentang Dosis Aman**: Berikan rentang dosis aman berdasarkan berat badan pasien (mg/kg/hari).

    Jawaban harus lengkap, menggunakan Bahasa Indonesia, dan menyertakan pemikiran/logika (Thoughts) dalam Bahasa Indonesia.
    Tidak lebih dari 160 kata. 
    Akhiri jawaban dengan 'Konfirmasi kembali sesuai panduan'.
    """
    try:
        response = llm.predict(query)  # Panggil GPT langsung
        return response
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"

hitungdosis = Tool(
    name="HitungDosisTool",
    func=hitungdosis_tool,
    description="Alat untuk mengambil informasi panduan dosis obat menggunakan Bahasa Indonesia, sehingga akan menghasilkan jawaban dengan BAHASA INDONESIA. Jawaban dibikin list untuk sirup, drops, puyer, dan dosis aman"
)

# Step 4: Inisialisasi Agent dengan Tools
tools = [hitungdosis]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Agent yang langsung merespons deskripsi alat
    verbose=True,
)

def calculate_rouge(reference: str, response: str):
    """
    Fungsi untuk menghitung skor ROUGE antara referensi dan respons.
    Args:
        reference (str): Teks referensi.
        response (str): Teks respons.
    Returns:
        dict: Skor ROUGE-1, ROUGE-2, dan ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, response)

    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }

# Initialize Streamlit app
st.set_page_config(page_title="Medical Chatbot", layout="wide")

# Sidebar
import streamlit as st

st.sidebar.image("gambar bot.jpg", width=150)
st.sidebar.title("Medical Chatbot")
st.sidebar.write("Pilih fitur yang ingin dipakai")

# Dropdown menu
feature = st.sidebar.selectbox("Pilih fitur:", ["QnA", "Hitung Dosis", "Evaluation Metrics"])

# Initialize session state
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "qa" not in st.session_state:
    st.session_state["qa"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "evaluation_metrics" not in st.session_state:
    st.session_state["evaluation_metrics"] = []

if feature == "QnA":
    st.subheader("Fitur QnA")
    st.sidebar.header("Upload Your Medical File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an Excel or PDF file", type=["xlsx", "pdf"], key="file_uploader_qna"
    )
    user_query = st.chat_input("Masukkan pertanyaan:", key="chat_input_qna")

    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file

        if st.session_state.get("qa") is None:
            try:
                # Load the database depending on the file type
                st.session_state["qa"] = load_db(uploaded_file, "stuff", 4)
                st.sidebar.success("Database loaded successfully!")
            except ValueError as e:
                st.sidebar.error(str(e))
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")

    if st.session_state.get("uploaded_file") and st.session_state.get("qa"):
        if user_query:
            # Process the query and maintain chat history
            chat_history = [
                (msg["role"], msg["content"])
                for msg in st.session_state["chat_history"]
            ]
            response = st.session_state["qa"](
                {"question": user_query, "chat_history": chat_history}
            )

            st.session_state["chat_history"].append(
                {"role": "user", "content": user_query}
            )

            if "tidak memiliki informasi" in response["answer"].lower():
                gpt_response = llm.predict(
                    f"Berikan penjelasan umum mengenai {user_query} tanpa konteks dokumen."
                )
                fallback_answer = f"Saya tidak menemukan informasi dalam dokumen. Berikut penjelasan umum:\n\n{gpt_response}"
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": fallback_answer}
                )
            else:
                # Extract and display source documents
                if "source_documents" in response:
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": response["answer"]}
                    )
                    
                    st.sidebar.write("Dokumen Sumber:")
                    sidebar_references = [
                        doc.page_content for doc in response["source_documents"]
                    ]
                    for ref in sidebar_references:
                        st.sidebar.markdown(f"- {ref[:200]}...")

                context = sidebar_references + [
                    msg["content"]
                    for msg in st.session_state["chat_history"]
                    if msg["role"] == "user"
                ]
                references = "\n".join(sidebar_references)

                evaluate_response_with_ragas(
                    references, context, response["answer"], user_query
                )

    # Display chat history
    if "chat_history" in st.session_state:
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.button("Clear History", key="clear_history_qna"):
        st.session_state["chat_history"] = []
        st.session_state["evaluation_metrics"] = []
        st.session_state["uploaded_file"] = None
        st.session_state["qa"] = None
        st.success("Chat history and uploaded file cleared!")

if feature == "Evaluation Metrics":
    st.subheader("Evaluation Metrics")
    if st.session_state["evaluation_metrics"]:
        metrics_df = pd.DataFrame(st.session_state["evaluation_metrics"])
        st.table(metrics_df)
    else:
        st.write("No evaluation metrics available. Ask a question in the QnA section first.")

if feature == "Hitung Dosis":
    st.subheader("Fitur Hitung Dosis")
    st.write("Masukkan informasi pasien dan obat untuk perhitungan dosis.")

    # Input terstruktur
    obat = st.text_input("Nama Obat:", key="nama_obat")
    usia = st.number_input("Usia Pasien (tahun):", min_value=0, max_value=120, step=1, key="usia")
    bb = st.number_input("Berat Badan Pasien (kg):", min_value=0.0, max_value=300.0, step=0.1, key="berat_badan")

    if st.button("Kirim Pertanyaan", key="submit_dosage"):
        if obat.strip() and usia > 0 and bb > 0:
            with st.spinner("Menghitung dosis, harap tunggu..."):
                try:
                    # Membuat input untuk alat
                    input_text = f"Nama Obat: {obat}\nUsia: {usia} tahun\nBerat Badan: {bb} kg"

                    # Panggil agent untuk menjalankan query
                    response = agent.run(input_text)

                    reference = """
                    Berikut adalah panduan dosis {obat} untuk anak berusia {usia} tahun dengan berat {bb} kg:

                    Sirup {obat}: Dosis yang direkomendasikan adalah 15 mg/kg per dosis. Untuk anak ini, dosisnya adalah 500 mg per dosis, yang setara dengan sekitar 4.17 ml sirup dengan konsentrasi 120 mg/ml. Diberikan setiap 4-6 jam sesuai kebutuhan.

                    Drops {obat}: Dosis yang dibutuhkan adalah 500 mg, yang setara dengan 5 ml atau sekitar 100 tetes (dengan asumsi 1 ml = 20 tetes). Frekuensi pemberian sama dengan sirup.

                    Puyer: Untuk membuat 10 puyer dengan dosis 500 mg per puyer, diperlukan 10 tablet paracetamol dengan dosis 500 mg per tablet.

                    Rentang Dosis Aman: Rentang dosis aman paracetamol adalah 10-15 mg/kg per dosis, dengan maksimum 60 mg/kg per hari. Untuk anak 40 kg, rentang dosis aman adalah 400-600 mg per dosis, dengan maksimum 2400 mg per hari.

                    Pastikan untuk selalu mengikuti rekomendasi dosis dan berkonsultasi dengan dokter atau apoteker sebelum memberikan obat.
                    """.format(obat=obat, usia=usia, bb=bb)

                    # Hitung skor ROUGE
                    rouge_scores = calculate_rouge(reference, response)

                    # Tampilkan hasil
                    st.write("Hasil Perhitungan Dosis")
                    st.markdown(response)
                    st.sidebar.header(f"**Hasil Evaluasi**")
                    st.sidebar.write(f"**Skor ROUGE-1:** {rouge_scores['ROUGE-1']:.2f}")
                    st.sidebar.write(f"**Skor ROUGE-2:** {rouge_scores['ROUGE-2']:.2f}")
                    st.sidebar.write(f"**Skor ROUGE-L:** {rouge_scores['ROUGE-L']:.2f}")

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
        else:
            st.error("Harap masukkan nama obat, usia, dan berat badan pasien dengan lengkap.")
