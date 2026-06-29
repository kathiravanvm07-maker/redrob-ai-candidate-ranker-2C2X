import pandas as pd
import os
import json
import csv
import re
import docx2txt
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Define strict filtering logic
STRICT_WHITELIST_RE = re.compile(
    r'\b(ai|ml|machine learning|search|retrieval|ranking|nlp|data scientist|information retrieval|llm)\b',
    re.IGNORECASE
)

BLACKLIST_RE = re.compile(
    r'\b(marketing|hr|accountant|support|operations|sales|analyst|product manager|scrum|manager|director|vp|head|lead|principal|c#|\\.net|asp\\.net|computer vision|opencv|yolo|image processing|segmentation|object detection|mechanical|civil|electrical|hardware)\b',
    re.IGNORECASE
)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def run_normalized_semantic_ranking():
    try:
        jd_content = docx2txt.process('/content/job_description.docx')
    except:
        print("JD file not found.")
        return

    candidates_to_embed = []
    with open('candidates.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                cand = json.loads(line)
                profile = cand.get('profile', {})
                title = profile.get('current_title', '')
                summary = profile.get('summary', '')
                
                if not STRICT_WHITELIST_RE.search(title):
                    continue
                if BLACKLIST_RE.search(title) or BLACKLIST_RE.search(summary):
                    continue
                if cand.get('redrob_signals', {}).get('recruiter_response_rate', 0.0) < 0.65:
                    continue

                skills = ", ".join([s.get('name', '') for s in cand.get('skills', [])])
                doc_text = f"{title} {summary} {skills}"
                candidates_to_embed.append({'data': cand, 'doc': Document(page_content=doc_text, metadata={"id": cand.get('candidate_id')})})
            except: continue

    if not candidates_to_embed: 
        print("No candidates matched criteria.")
        return

    docs = [c['doc'] for c in candidates_to_embed]
    vector_db = FAISS.from_documents(docs, embeddings)
    # Similarity search with relevance scores
    relevant_docs = vector_db.similarity_search_with_relevance_scores(jd_content, k=len(candidates_to_embed))
    
    if not relevant_docs:
        return

    # Normalize scores to ensure they are out of 1.0 (Min-Max Scaling based on results)
    raw_scores = [score for doc, score in relevant_docs]
    max_s = max(raw_scores)
    min_s = min(raw_scores)
    
    # Re-calculate to make the top score close to 1.0 if it isn't already
    def normalize(s):
        if max_s == min_s: return 1.0
        return (s - min_s) / (max_s - min_s)

    # Export top 100 results
    output_csv = 'submission_semantic.csv'
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['candidate_id', 'semantic_match_score', 'summary'])
        for doc, score in relevant_docs[:100]:
            cid = doc.metadata['id']
            # Apply normalization to scale towards 1.0
            norm_score = normalize(score)
            writer.writerow([cid, round(norm_score, 4), doc.page_content[:200]])

    df_results = pd.read_csv(output_csv)
    print(f"Extracted {len(df_results)} candidates with normalized scores.")
    display(df_results.head())

run_normalized_semantic_ranking()
