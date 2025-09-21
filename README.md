# 💼 SmartSpend Dashboard

**SmartSpend** is an interactive **personal finance dashboard** built with **Streamlit**. It helps users analyze, visualize, and forecast their expenses while providing an **offline AI assistant** powered by **LlamaIndex + Ollama**, with optional GPT-based fallback.

## Features
### 📊 Dashboard
- Upload your personal finance Excel file and analyze transactions.
- Filter by date range, category, and transaction type (Income/Expense).
- Set monthly budgets per category and track overspending.
- Visualize:
  - Income vs Expense distribution (Pie chart)
  - Spending trends (Line chart)
  - Category-wise breakdown (Bar chart)
- Forecast future expenses using Exponential Smoothing (Holt-Winters method).

### 🤖 SmartSpend Assistant
- Ask questions about your finances using a local LlamaIndex + Ollama model.
- Optionally fallback to:
  - OpenAI GPT-3.5
  - HuggingFace Falcon-7B
- Intelligent responses with weak-answer detection to ensure fallback triggers if needed.

---

## Setup
1. Clone the repository:
```bash
git clone https://github.com/meetbhayani/SmartSpent.git
cd SmartSpent
```
2. Create a .env file with your API keys (if using GPT fallback):
```bash
OPENAI_API_KEY=your_openai_api_key
```

3. Run **Ollama** with a local model like gemma:2b.

4. Launch the Streamlit app:
```bash
streamlit run smartspent_dashboard.py
```

## Usage
1. Upload your Excel file with personal finance transactions.
2. Use sidebar filters to select date ranges, categories, or transaction types.
3. Set your monthly budgets per category.
4. Visualize insights and track your spending.
5. Ask the SmartSpend Assistant any finance-related questions.
6. Optionally, enable GPT fallback if local responses are unclear.
