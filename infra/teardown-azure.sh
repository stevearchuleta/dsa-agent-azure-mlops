#!/usr/bin/env bash
# ============================================================
# Azure Teardown - Deletes ALL resources to stop billing
# ============================================================
set -euo pipefail

RESOURCE_GROUP="rg-dsa-agent"

echo "WARNING: This will delete resource group '' and ALL resources inside it."
echo "Press Ctrl+C within 10 seconds to cancel..."
sleep 10

az group delete --name $RESOURCE_GROUP --yes --no-wait
echo "Deletion initiated. Resources will be removed within a few minutes."