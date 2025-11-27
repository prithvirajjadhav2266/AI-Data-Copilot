# AI Data Copilot

Gemini-powered notebook + dashboard for cleaning datasets, running natural-language transformations, and generating Plotly visualizations.

## 1. Prerequisites
- Python 3.10+ (3.11 recommended)
- Git
- Google Gemini API key with access to text + code models (e.g., `models/gemini-2.5-flash`)
- Node not required; frontend uses Streamlit.

## 2. Set up locally
```powershell
git clone https://github.com/prithvirajjadhav2266/AI-Data-Copilot.git
cd AI-Data-Copilot
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure environment
Create a user or session variable for your Gemini key (replace `<key>`):
```powershell
setx GEMINI_API_KEY "<key>"
# reopen terminal, then:
$Env:GEMINI_MODEL = "models/gemini-2.5-flash"
$Env:GEMINI_TEMPERATURE = "0.1"
```
(You can skip `setx` and only export in-session if preferred. Never commit the key.)

## 4. Run services
Open two terminals (both with the venv activated).

**Backend**
```powershell
python -m uvicorn backend.app:app --reload
```

**Frontend**
```powershell
streamlit run frontend/app.py
```

Visit http://localhost:8501.

## 5. Using the app
1. Upload a CSV via the left sidebar.  
2. Explore descriptive profile in “Dataset Snapshot.”  
3. Use **Gemini Cleaning Plan** for structured fixes (fill missing, drop duplicates, clip outliers).  
4. Use **Gemini Code Agent** for free-form pandas transformations; review code, dry-run, then apply.  
5. Use **Natural Language to Visualization** to request charts; inspect the plan, execute, and view Plotly output and explanation.  
6. Download the modified dataset or undo to prior snapshots through the dataset history panel.

All LLM responses are validated (schema, allowed actions, column checks) and logged. Reject or edit plans before applying.

## 6. Testing
```powershell
python -m pytest
```

## 7. Deployment notes
- Keep `.env`, `.venv/`, and API keys out of source control (already covered in `.gitignore`).  
- For production, supply environment variables through your orchestrator and disable Streamlit’s development options.  
- Adjust logging/Audit sinks under `backend/storage.py` if you need persistent history.

## 8. Troubleshooting
- **Gemini 404**: ensure `GEMINI_MODEL` points to a model available to your key.  
- **Gemini MAX_TOKENS**: retry; backend automatically re-prompts with trimmed context.  
- **Import errors in generated code**: sandbox already exposes `pd`, `np`, `px`, `go`; avoid adding imports.  
- **NaN serialization errors**: ensure backend is the latest version (sanitization is built in).

Enjoy exploring datasets with natural language control.