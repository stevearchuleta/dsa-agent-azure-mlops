# DSA Agent Azure MLOps Evidence

## Project status

This project demonstrates a data science agent package with deterministic analysis tools, Azure OpenAI configuration, Azure ML pipeline execution, GitHub Actions CI/CD, GitHub OIDC authentication, and Azure Container Registry image publishing.

## Proven milestones

| Area | Evidence |
|---|---|
| Latest main commit | 7e89bc4 Add ACR image build workflow |
| Azure OpenAI chat deployment | gpt-4o-mini-dsa |
| Azure OpenAI embedding deployment | text-embedding-3-small-dsa |
| Azure ML workspace | aml-dsa-agent-eastus-001 |
| Azure ML data asset | dsa-agent-sample-data:1 |
| Azure ML GitHub Actions job | helpful_nutmeg_cmd5lpl73b completed |
| Azure ML evaluation report | Overall passed: True |
| ACR image repository | dsa-agent |
| ACR image tags | latest, sha-7e89bc4 |
| GitHub Actions OIDC login | Proven by Azure OIDC Login Smoke Test |
| GitHub Actions Azure ML workflow | Validation and submit modes proven |
| GitHub Actions ACR workflow | Build and push modes proven |

## Deterministic tool coverage

| Tool area | Module | Purpose |
|---|---|---|
| EDA | src/dsa/agents/eda_tools.py | DataFrame overview, missing values, numeric and categorical summaries |
| Plotting | src/dsa/agents/plot_agent.py | Deterministic matplotlib plots |
| Statistics | src/dsa/agents/stats_agent.py | Deterministic hypothesis testing |
| Machine learning | src/dsa/agents/ml_agent.py | Deterministic classifier training and evaluation |
| SQL | src/dsa/agents/sql_agent.py | Read-only SQLite/SQLAlchemy querying |
| RAG | src/dsa/rag | FAISS-based document retrieval foundation |

## Safety controls

- Local secrets are stored in config/config.local.json.
- config/config.local.json is git-ignored.
- The safety scanner blocks tracked or commit-candidate secret-like values.
- GitHub Actions uses OIDC instead of long-lived Azure client secrets.
- Azure resource group budget exists for monthly cost visibility.
- Local outputs and downloaded Azure ML artifacts are ignored by Git.

## Public repository note

The repository is public. Real local credentials, downloaded outputs, and generated local databases must remain outside Git tracking.

## Deferred improvements

1. Add an interactive router or CLI demo that chooses among deterministic tools.
2. Rename *_agent.py modules to *_tools.py during the router refactor.
3. Disable ACR admin user after re-proving image push and Azure ML workflows.
4. Decide whether to keep the old GHCR tag workflow or consolidate fully on ACR.
5. Expand README screenshots and portfolio narrative.
