import os
import sys

# Add parent directory to path so backend_server can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_server import app
