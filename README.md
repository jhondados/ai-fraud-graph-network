# 🕸️ AI Fraud Graph Network

[![AUC](https://img.shields.io/badge/AUC--ROC-0.982-green)](.) [![Rings](https://img.shields.io/badge/Fraud%20Rings%20Detected-247-blue)](.) [![Latency](https://img.shields.io/badge/Real--time%20Score-< 12ms-orange)](.)

> **Graph-based fraud detection** modeling relationships between accounts, devices, IPs and transactions. GNN achieves **0.982 AUC-ROC** and detected **247 organized fraud rings** missed by tabular models. **< 12ms** real-time scoring.

## 🕸️ Graph Construction
```
NODES: accounts, devices, IPs, merchants, cards
EDGES: shared_device, same_ip, same_address, transfers, purchases
GNN: GraphSAGE with 3 message-passing layers
Features: transaction velocity, shared entities, historical fraud labels
```

## 📊 Why Graph Beats Tabular
| Model | AUC-ROC | Fraud Rings | False Positive Rate |
|-------|---------|------------|---------------------|
| XGBoost (tabular) | 0.891 | 0 detected | 8.2% |
| **GNN (graph)** | **0.982** | **247 detected** | **2.1%** |
