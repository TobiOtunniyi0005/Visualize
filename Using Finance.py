python -m venv venv
.\venv\Scripts\activate
pip install scikit-learn pandas yfinance streamlit plotly
pip uninstall pandas-ta -y
streamlit run Finance.py