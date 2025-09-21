# SmartSpend Dashboard with Local (Offline) RAG Assistant using LlamaIndex + Ollama

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from dotenv import load_dotenv

# --- LlamaIndex with Ollama ---
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SimpleNodeParser

# --- Setup ---
load_dotenv()
st.set_page_config(page_title="SmartSpend Dashboard", page_icon="📊", layout="wide")

# --- Title ---
st.markdown("""
<div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 12px; text-align: center;">
    <h1 style="color: #b9f6fb; font-size: 36px;">💼 SmartSpend: Welcome User</h1>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload your personal finance Excel file", type=["xlsx"])

@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_excel(file, sheet_name="Sheet1", parse_dates=["Date / Time"])
    df = df.rename(columns={
        "Date / Time": "Date",
        "Income/Expense": "Type",
        "Debit/Credit": "Amount",
        "Sub category": "SubCategory",
    })
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Description"] = df.apply(lambda r: f"{r['Mode']} - {r['SubCategory']} - ₹{r['Amount']} on {r['Category']}", axis=1)
    return df

def df_to_documents(df):
    grouped = df.groupby(df["Date"].dt.to_period("M"))
    raw_docs = [
        Document(text=f"{month} Transactions:\n" + "\n".join(
            f"- {row['Date'].date()}: {row['Type']} of ₹{row['Amount']} for {row['SubCategory']} ({row['Category']}) via {row['Mode']}"
            for _, row in group.iterrows()
        )) for month, group in grouped
    ]
    return raw_docs

@st.cache_resource
def build_llamaindex_engine(_docs):
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core import VectorStoreIndex

    # Define Ollama LLM (you must have ollama running with a model like llama3)
    llm = Ollama(model="gemma:2b",request_timeout=120)

    # Embeddings
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # Chunking documents
    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(_docs)

    # Build index
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    # ✅ This method works on ALL versions
    query_engine = index.as_query_engine(llm=llm)

    return query_engine


def is_weak_response(response):
    if not response.strip():
        return True
    return any(p in response.lower() for p in ["i don't know", "not sure", "no relevant", "unable", "not enough information"])

if uploaded_file:
    df = load_data(uploaded_file)

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filter Options")
    min_date, max_date = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    category_filter = st.sidebar.multiselect("Select categories", options=df["Category"].unique(), default=df["Category"].unique())
    type_filter = st.sidebar.radio("Transaction type", options=["All", "Income", "Expense"], index=0)

    filtered_df = df[
        (df["Date"] >= pd.to_datetime(date_range[0])) &
        (df["Date"] <= pd.to_datetime(date_range[1])) &
        (df["Category"].isin(category_filter))
    ]
    if type_filter != "All":
        filtered_df = filtered_df[filtered_df["Type"] == type_filter]

    # --- GPT Fallback Toggle ---
    use_fallback = st.sidebar.checkbox("Enable GPT Fallback", value=True)
    fallback_model = st.sidebar.selectbox("Fallback Model", ["OpenAI GPT-3.5", "HuggingFace Falcon-7B"], index=0)

    # --- Budgeting ---
    st.subheader("💰 Budgeting Dashboard")
    st.markdown("#### Set monthly budget per category:")
    categories = filtered_df["Category"].unique()
    budget_dict = {cat: st.number_input(cat, min_value=0, value=1000, step=500, key=f"budget_{cat}") for cat in categories}

    st.markdown("---")
    st.subheader("📊 Monthly Summary vs Budget")
    monthly_summary = filtered_df.groupby(["Month", "Category"])["Amount"].sum().unstack().fillna(0)
    latest_month = monthly_summary.index.max()
    if latest_month:
        st.markdown(f"#### Spending in {latest_month}")
        for cat in monthly_summary.columns:
            spent = monthly_summary.loc[latest_month, cat]
            budget = budget_dict.get(cat, 0)
            percent = (spent / budget) * 100 if budget else 0
            st.markdown(f"**{cat}**: ₹{spent:.2f} / ₹{budget:.2f}")
            st.progress(min(100, int(percent)), text=f"{percent:.1f}% used")

        overspent = [(cat, spent, budget_dict[cat]) for cat, spent in monthly_summary.loc[latest_month].items() if budget_dict.get(cat, 0) and spent > budget_dict[cat]]
        if overspent:
            st.markdown("#### You overspent in:")
            for cat, spent, budget in overspent:
                st.warning(f"💸 *{cat}*: Overspent by ₹{spent - budget:.2f}")
        else:
            st.success("✅ Great job! You stayed within budget.")

    # --- Visualizations ---
    st.markdown("---")
    st.subheader("🍰 Income vs Expense Distribution")
    totals = filtered_df.groupby("Type")["Amount"].sum()
    if not totals.empty:
        st.plotly_chart(px.pie(values=totals.values, names=totals.index, title="Income vs Expense"), use_container_width=True)

    st.subheader("📈 Spending Trend")
    trend = filtered_df[filtered_df["Type"] == "Expense"].groupby("Month")["Amount"].sum().reset_index()
    st.plotly_chart(px.line(trend, x="Month", y="Amount", title="Monthly Expenses", markers=True), use_container_width=True)

    st.subheader("📋 Category Breakdown")
    cat_break = filtered_df[filtered_df["Type"] == "Expense"].groupby("Category")["Amount"].sum().sort_values()
    st.bar_chart(cat_break)

    st.subheader("📅 Forecasted Expenses")
    monthly_exp = filtered_df[filtered_df["Type"] == "Expense"].groupby("Month")["Amount"].sum().sort_index()
    if len(monthly_exp) >= 6:
        forecast = ExponentialSmoothing(monthly_exp, seasonal="add", seasonal_periods=3).fit().forecast(3)
        st.line_chart(forecast)
    else:
        st.warning("⚠️ Need at least 6 months of data for forecasting.")

    # --- 🤖 SmartSpend Assistant ---
    st.markdown("---")
    st.subheader("🤖 Ask SmartSpend Assistant")

    raw_docs = df_to_documents(df)
    query_engine = build_llamaindex_engine(raw_docs)
    user_q = st.text_input("Ask anything about your finances or personal budgeting...")

    if user_q:
        with st.spinner("Thinking locally..."):
            local_response = str(query_engine.query(user_q))

        if is_weak_response(local_response) and use_fallback:
            st.warning("🧠 Local model unclear. Trying fallback...")
            try:
                if fallback_model == "OpenAI GPT-3.5":
                    import openai
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful finance assistant."},
                            {"role": "user", "content": user_q}
                        ],
                    )
                    fallback_response = completion['choices'][0]['message']['content']
                else:
                    from transformers import pipeline
                    pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", trust_remote_code=True)
                    fallback_response = pipe(user_q, max_length=512, do_sample=True)[0]['generated_text']
                st.success(fallback_response)
            except Exception as e:
                st.error(f"⚠️ Fallback failed: {e}")
        else:
            st.success(local_response)

    if st.button("Reset"):
        st.session_state.user_question = ""

    st.markdown("---")
    st.markdown("<h4 style='text-align: center;'>👋 Thank you for using <b>SmartSpend</b>! Have a great day!</h4>", unsafe_allow_html=True)
