#!/usr/bin/env bash
# ============================================================
# Azure ML Workspace Setup
# Run ONCE to create resources. Costs ~-2/month when idle.
# ============================================================
# Prerequisites:
#   1. az login
#   2. az extension add -n ml
# ============================================================

set -euo pipefail

# ---------- Configuration ----------
RESOURCE_GROUP="rg-dsa-agent"
LOCATION="westus2"
WORKSPACE="mlw-dsa-agent"
BUDGET_NAME="dsa-budget"
BUDGET_LIMIT=5          # USD per month

# ---------- Resource Group ----------
echo "[1/4] Creating resource group..."
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION

# ---------- Budget Alert ----------
echo "[2/4] Creating budget alert ($$BUDGET_LIMIT/month)..."
az consumption budget create \
  --budget-name $BUDGET_NAME \
  --resource-group $RESOURCE_GROUP \
  --amount $BUDGET_LIMIT \
  --time-grain Monthly \
  --start-date $(date -d "first day of this month" +%Y-%m-01) \
  --end-date 2027-12-31 \
  --category Cost

# ---------- ML Workspace ----------
echo "[3/4] Creating ML workspace (this takes 2-3 minutes)..."
az ml workspace create \
  --name $WORKSPACE \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# ---------- Verify ----------
echo "[4/4] Verifying workspace..."
az ml workspace show \
  --name $WORKSPACE \
  --resource-group $RESOURCE_GROUP \
  --query "{name:name, location:location, state:provisioning_state}" \
  --output table

echo ""
echo "Done! Workspace ready. Budget alert set at $$BUDGET_LIMIT/month."
echo "To tear down: az group delete --name $RESOURCE_GROUP --yes --no-wait"