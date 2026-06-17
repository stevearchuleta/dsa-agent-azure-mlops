# %% [markdown]
# # DSA Agent - Demo Notebook
# Thin orchestrator: all logic lives in the dsa package.

# %% Cell 1 - Configuration
from dsa.config import ensure_api_key, PAPERS_DIR, FAISS_DIR, ARTIFACTS_DIR

api_key = ensure_api_key()
print(f"Papers dir:    {PAPERS_DIR}")
print(f"FAISS dir:     {FAISS_DIR}")
print(f"Artifacts dir: {ARTIFACTS_DIR}")

# %% Cell 2 - Build or load FAISS index
from dsa.rag.ingest import get_or_build_index

index = get_or_build_index()
print(f"Index ready: {index.index.ntotal} vectors")

# %% Cell 3 - RAG chain
from dsa.rag.chain import build_rag_chain

rag_chain = build_rag_chain()
answer = rag_chain.invoke("What are the main findings of the paper?")
print(answer)

# %% Cell 4 - Pandas EDA agent
import pandas as pd
from dsa.agents.pandas_agent import build_pandas_agent

# Replace with your actual DataFrame
df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "price": range(100)})
agent = build_pandas_agent(df)
result = agent.invoke("What is the average price?")
print(result)

# %% Cell 5 - Export artifacts
from dsa.export.artifacts import save_figure, save_table
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(df["date"], df["price"])
ax.set_title("Sample Price Series")

fig_path = save_figure(fig, "demo_price_plot.png")
tbl_path = save_table(df.describe(), "demo_summary_stats.csv")
print(f"Figure: {fig_path}")
print(f"Table:  {tbl_path}")