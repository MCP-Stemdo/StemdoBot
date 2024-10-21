py -3.10 -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt

uvicorn app:app --reload --host 0.0.0.0 --port 8501 
streamlit run app_streamlit.py


para py 3.11 es :Activate .venv
deactivate 
Remove-Item -Recurse -Force .venv
eliminar __pycache__
------------------
documentación: estándares - PEP 257
