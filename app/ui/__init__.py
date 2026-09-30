"""Streamlit UI package for Vestige (legacy UI).

`app/streamlit_app.py` is the entry point (page config, bootstrap, navigation). Everything the
pages share lives here: guarded execution of long or LLM-backed steps (`guard`), the on-disk
case store (`cases`), the per-session engine and analysis-state helpers (`runtime`), shared
frames, badges and the status strip (`components`), and one module per page (`pages`).
"""
