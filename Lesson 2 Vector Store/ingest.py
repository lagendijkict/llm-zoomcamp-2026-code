import json, requests
from minsearch import Index

def load_faq_data():
    # Local file, because json overview file was no longer available
    docs_path = '/workspaces/llm-zoomcamp-2026-code/Lesson 1 RAG/sources/courses.json'
    with open(docs_path, 'r') as f:
        courses_raw = json.load(f)
     
    documents = []
    url_prefix = 'https://datatalks.club/faq'   
    
    for course in courses_raw:
        course_url = f'{url_prefix}{course['path']}'

        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()

        documents.extend(course_data)
            
    return documents

def build_index(documents):
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"]
    )
    index.fit(documents)
    return index
