from pathlib import Path
import runpy
import sys
import traceback

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
APP_FILE = BASE_DIR / "app" / "app.py"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    # Execute the Streamlit entry file by path to avoid package-name collisions
    # on hosted environments where `app` may resolve differently.
    runpy.run_path(str(APP_FILE), run_name="__main__")
except Exception as exc:  # pragma: no cover - deployment-facing fallback
    try:
        st.set_page_config(page_title="Xente Loan Default App", page_icon="⚠️", layout="wide")
    except Exception:
        pass
    st.title("Xente Loan Default App")
    st.error("The application could not start successfully.")
    st.info(
        "If this is Streamlit Community Cloud, confirm that the repository includes the saved model, "
        "generated outputs, and all required dependencies."
    )
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
    st.stop()
