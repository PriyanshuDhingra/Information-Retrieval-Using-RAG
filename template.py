import os
from pathlib import Path

list_of_files = [
    "QAWithPdf/__init__.py",
    "QAWithPdf/data_ingestion.py",
    "QAWithPdf/embedding.py",
    "QAWithPdf/model_api.py",
    "Experiments/experiments.ipynb",
    "StreamlitApp.py",
    "logger.py",
    "exception.py",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass