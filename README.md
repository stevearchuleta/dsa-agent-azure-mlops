# DSA Agent Azure MLOps

An Azure MLOps data science agent project that combines deterministic data science tools, Azure OpenAI configuration, Azure Machine Learning pipelines, GitHub Actions CI/CD, GitHub OIDC authentication, and Azure Container Registry image publishing.

## What this project demonstrates

- Azure OpenAI chat and embedding deployments.
- Deterministic EDA, plotting, statistics, machine learning, SQL, and RAG-oriented package modules.
- Azure ML pipeline components and pipeline job YAML.
- GitHub Actions CI for linting and tests.
- GitHub Actions OIDC login to Azure without long-lived Azure secrets.
- Manual GitHub Actions workflow for Azure ML pipeline validation and submission.
- Manual GitHub Actions workflow for Docker image build and push to Azure Container Registry.
- Local secret safety scanner and Git ignore guardrails.

## Architecture

```text
Local package code
  src/dsa/
    config.py
    llm.py
    embeddings.py
    agents/
    rag/

CI/CD
  .github/workflows/ci.yml
  .github/workflows/oidc-login-smoke.yml
  .github/workflows/azure-ml.yml
  .github/workflows/build-image.yml

Azure ML
  mlops/azureml/components/
  mlops/azureml/pipelines/
  mlops/azureml/scripts/

Security
  scripts/02_verify_project_safety.ps1
  config/config.local.json ignored locally
```

## Current proof points

| Capability | Status |
|---|---|
| Main branch | 7e89bc4 Add ACR image build workflow |
| Safety scanner | Passing |
| GitHub Actions CI | Passing |
| Azure OIDC login | Passing |
| Azure ML workflow validation | Passing |
| Azure ML workflow real submission | Passing |
| Azure ML evaluation report | Overall passed: True |
| Docker image build | Passing |
| ACR image push | Passing |
| ACR image tags | latest, sha-7e89bc4 |

See docs/PROJECT_EVIDENCE.md for milestone details.

## Local setup

```powershell
conda activate dsa-agent-azure-mlops
python -m pip install -e ".[dev]"
python -m ruff check src tests scripts mlops
python -m pytest tests -v
.\scripts\02_verify_project_safety.ps1
```

## Secret handling

Real credentials are not committed.

Local Azure OpenAI credentials live in:

```text
config/config.local.json
```

That file is ignored by Git. The repository includes safe examples only.

## Azure resources used

| Resource | Name |
|---|---|
| Resource group | rg-dsa-agent-mlops-eastus-001 |
| Azure ML workspace | aml-dsa-agent-eastus-001 |
| Azure OpenAI resource | aoai-dsa-agent-eastus-001 |
| Chat deployment | gpt-4o-mini-dsa |
| Embedding deployment | text-embedding-3-small-dsa |
| Azure ML data asset | dsa-agent-sample-data:1 |
| Azure Container Registry | 0fceba56009846a1bc590f00b9de6ba0.azurecr.io |

## GitHub Actions workflows

| Workflow | Purpose |
|---|---|
| CI | Lint and test package code |
| Azure OIDC Login Smoke Test | Prove GitHub Actions can log in to Azure using OIDC |
| Azure ML Pipeline | Validate or submit the Azure ML pipeline |
| Build Image to ACR | Build and optionally push the Docker image to ACR |

## Project status

The MLOps foundation is complete. The next phase is an interactive router or CLI demo that chooses among the deterministic data science tools.

## Interactive router CLI

Session 8 adds a deterministic router that classifies every question first,
then dispatches only when the selected tool has the required inputs.

```powershell
python -m dsa.cli "show missing values and dataframe overview"
python -m dsa.cli "plot a histogram by target"
python -m dsa.cli --json "SQL: select * from employees" --db .\sample.sqlite
python -m dsa.cli --list-routes
```

After editable installation, the console script is also available:

```powershell
dsa-agent "train a random forest classifier"
```
