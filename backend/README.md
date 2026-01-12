# Backend – Claim Copilot API

## Setup
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m app.ml.train_text_model
uvicorn app.main:app --reload
```

API runs at http://127.0.0.1:8000
