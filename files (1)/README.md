# DevelopersHub Corporation — AI/ML Engineering Internship
## Advanced Tasks Portfolio

> **Deadline**: April 28, 2026 · **Completed**: 5/5 tasks

---

## 📁 Repository Structure

```
internship/
├── task1_bert_classifier/          ← Task 1: BERT News Classifier
│   ├── train.py                    #   Fine-tuning script
│   ├── app.py                      #   Streamlit deployment
│   ├── requirements.txt
│   └── Task1_BERT_News_Classifier.ipynb
│
├── task2_churn_pipeline/           ← Task 2: ML Pipeline – Customer Churn
│   ├── pipeline.py                 #   Full sklearn pipeline + GridSearch
│   ├── requirements.txt
│   └── Task2_Churn_ML_Pipeline.ipynb
│
├── task3_multimodal_housing/       ← Task 3: Multimodal Housing Prediction
│   ├── multimodal.py               #   CNN + Tabular fusion model
│   ├── requirements.txt
│   └── Task3_Multimodal_Housing.ipynb
│
├── task4_rag_chatbot/              ← Task 4: RAG Context-Aware Chatbot
│   ├── app.py                      #   Streamlit chatbot UI
│   ├── rag_engine.py               #   LangChain + ChromaDB pipeline
│   ├── corpus/                     #   Knowledge base documents
│   └── Task4_RAG_Chatbot.ipynb
│
└── task5_auto_tag_tickets/         ← Task 5: Auto-Tag Support Tickets
    ├── auto_tag.py                 #   LLM tagging with strategy comparison
    ├── requirements.txt
    └── Task5_Auto_Tag_Tickets.ipynb
```

---

## Task Summaries

### Task 1 — News Topic Classifier Using BERT
**Model**: `bert-base-uncased` fine-tuned on AG News (4 classes)  
**Approach**: Hugging Face Transformers + custom PyTorch training loop  
**Results**: ~94% accuracy · ~94% weighted F1  
**Deploy**: `streamlit run task1_bert_classifier/app.py`

```bash
cd task1_bert_classifier
pip install -r requirements.txt
python train.py           # Fine-tune (saves bert_ag_news.pt)
streamlit run app.py      # Launch UI
```

---

### Task 2 — End-to-End ML Pipeline (Customer Churn)
**Model**: Logistic Regression + Random Forest via sklearn `Pipeline`  
**Approach**: ColumnTransformer preprocessing + GridSearchCV tuning + joblib export  
**Results**: RF AUC ~0.84 · Tuned LR AUC ~0.82  
**Export**: `churn_pipeline.joblib`, `rf_pipeline.joblib`

```bash
cd task2_churn_pipeline
pip install -r requirements.txt
python pipeline.py
```

---

### Task 3 — Multimodal Housing Price Prediction
**Model**: MobileNetV2 CNN + Tabular MLP → Fusion head  
**Approach**: Image features + 6 tabular features, Huber loss, AdamW  
**Metrics**: MAE and RMSE on held-out test set  
**Dataset**: Procedurally generated (1200 synthetic samples)

```bash
cd task3_multimodal_housing
pip install -r requirements.txt
python multimodal.py
```

---

### Task 4 — Context-Aware RAG Chatbot *(previously completed)*
**Stack**: LangChain + ChromaDB + `all-MiniLM-L6-v2` + Anthropic Claude  
**Approach**: MMR vector retrieval + ConversationBufferWindowMemory  
**Topics**: AI, ML, Deep Learning, NLP, Python

```bash
cd task4_rag_chatbot
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

---

### Task 5 — Auto-Tag Support Tickets Using LLM
**Model**: Claude (`claude-haiku-4-5`) via Anthropic API  
**Approach**: Zero-shot vs Few-shot vs Chain-of-Thought comparison  
**Output**: Top-3 tags with confidence scores per ticket  
**Dataset**: 30 tickets across 10 support categories

```bash
cd task5_auto_tag_tickets
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
python auto_tag.py
```

---

## Key Results Summary

| Task | Model | Key Metric |
|------|-------|-----------|
| 1 – News Classifier | BERT fine-tuned | ~94% F1 |
| 2 – Churn Pipeline | Random Forest | AUC ~0.84 |
| 3 – Housing Price | CNN + Tabular Fusion | MAE < $30K |
| 4 – RAG Chatbot | Claude + ChromaDB | Context-grounded answers |
| 5 – Ticket Tagging | Claude (few-shot) | Top-3 acc ~90% |

---

## Skills Demonstrated
- NLP with Transformers (BERT fine-tuning, transfer learning)
- Production ML Pipelines (sklearn, joblib, GridSearchCV)
- Multimodal Deep Learning (CNN + tabular feature fusion)
- Conversational AI (RAG, LangChain, vector stores)
- Prompt Engineering (zero-shot, few-shot, chain-of-thought)
- Deployment (Streamlit, Gradio-ready)
