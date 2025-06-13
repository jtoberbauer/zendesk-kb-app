import requests
import json
from bs4 import BeautifulSoup

# CONFIG: Replace with your details
ZENDESK_SUBDOMAIN = "basicops"  # e.g. "mycompany"
EMAIL = "team@basicops.com"            # append /token if using API token
API_TOKEN = "gmuvQ14SvfPFv63nViRHJ6S7qnDxZynRf9aBbMt8"

# Setup
BASE_URL = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com"
HEADERS = {"Content-Type": "application/json"}
AUTH = (EMAIL, API_TOKEN)

def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n").strip()

def fetch_articles():
    articles = []
    url = f"{BASE_URL}/api/v2/help_center/en-us/articles.json?page[size]=100"
    
    while url:
        response = requests.get(url, auth=AUTH, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        for item in data.get("articles", []):
            cleaned_body = clean_html(item["body"])
            articles.append({
                "id": item["id"],
                "title": item["title"],
                "section_id": item["section_id"],
                "updated_at": item["updated_at"],
                "html": item["body"],
                "content": cleaned_body,
                "url": item["html_url"]
            })
        url = data.get("next_page")
    
    return articles

if __name__ == "__main__":
    print("Fetching articles from Zendesk...")
    try:
        articles = fetch_articles()
        with open("zendesk_kb_articles.json", "w") as f:
            json.dump(articles, f, indent=2)
        print(f"✅ Exported {len(articles)} articles to 'zendesk_kb_articles.json'")
    except Exception as e:
        print(f"❌ Error: {e}")
