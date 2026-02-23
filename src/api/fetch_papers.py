import requests
import time


BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_cs_papers(limit=100):
    papers = []

    params = {
        "query": "Computer Science",
        "limit": 1,
        "fields": "paperId,title,abstract,year,citationCount,referenceCount,fieldsOfStudy"
    }

    for offset in range(0, limit, 50):
        params["offset"] = offset

        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print("Error:", response.status_code)
            break

        data = response.json()

        for paper in data.get("data", []):
            if "Computer Science" in (paper.get("fieldsOfStudy") or []):
                papers.append(paper)

        time.sleep(1)  # avoid rate limit

    return papers


if __name__ == "__main__":
    results = fetch_cs_papers(limit=100)
    print(f"Fetched {len(results)} papers")