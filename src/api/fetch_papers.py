import requests
import time
from dotenv import load_dotenv
import os

load_dotenv("src/config/.env")

API_KEY = os.getenv("S2_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Check your .env file.")

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

headers = {
    "x-api-key": API_KEY
}

def fetch_cs_papers(limit=100):
    papers = []
    batch_size = 50

    params = {
        "query": "machine learning OR computer systems OR databases",
        "fields": "paperId,title,abstract,year,citationCount,referenceCount,fieldsOfStudy"
    }

    for offset in range(0, limit, batch_size):
        params["limit"] = min(batch_size, limit - offset)
        params["offset"] = offset

        response = requests.get(BASE_URL, headers=headers, params=params)

        if response.status_code != 200:
            print("Error:", response.status_code)
            print(response.text)
            break

        data = response.json()

        for paper in data.get("data", []):
            if "Computer Science" in (paper.get("fieldsOfStudy") or []):
                papers.append(paper)

        print(f"Fetched batch starting at offset {offset}")
        time.sleep(1)

    return papers


if __name__ == "__main__":
    results = fetch_cs_papers(limit=100)
    print(f"Fetched {len(results)} papers")