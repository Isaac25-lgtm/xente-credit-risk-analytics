from pathlib import Path
import runpy
import sys


BASE_DIR = Path(__file__).resolve().parent
APP_FILE = BASE_DIR / "app" / "app.py"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Execute the Streamlit entry file by path to avoid package-name collisions
# on hosted environments where `app` may resolve differently.
runpy.run_path(str(APP_FILE), run_name="__main__")
