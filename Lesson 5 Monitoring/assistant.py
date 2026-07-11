import sys

from db_save import save_conversation
from dotenv import load_dotenv
from ingest import build_index, load_faq_data
from metrics import RAGWithMetrics
from openai import OpenAI
from rag_helper import RAGBase

load_dotenv()

def create_assistant():
    documents = load_faq_data()
    index = build_index(documents)
    
    return RAGBase(index=index,
                   llm_client=OpenAI())
    
if __name__ == "__main__":
    assistant = create_assistant()

    query = "How do I join the course?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    print(answer)

    save_conversation(assistant.last_call, query, "llm-zoomcamp")