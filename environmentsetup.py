import docx2txt
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. Extract Job Description Knowledge
def get_jd_text(file_path):
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Initialize Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

jd_text = get_jd_text('/content/job_description.docx')
if jd_text:
    print("Job Description Loaded and Embedded successfully.")
else:
    print("Failed to load Job Description. Please ensure the file exists at /content/job_description.docx")
