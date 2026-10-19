
import re
import requests

from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


SEED_URLS = [
    "https://www.iitdh.ac.in/",
    "https://www.iitdh.ac.in/academic-announcement",
    "https://www.iitdh.ac.in/academic-calendar",
    "https://www.iitdh.ac.in/academic-forms",
]

MAX_PAGES = 12

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "when",
    "what", "where", "how", "does", "do", "for", "of", "to",
    "in", "on", "at", "and", "or", "my", "me", "please",
    "i", "we", "you", "it", "this", "that", "be", "begin",
    "begins", "start", "starts", "from", "with", "about"
}


def clean_url(url):
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return None

    if parsed.hostname not in ("iitdh.ac.in", "www.iitdh.ac.in"):
        return None

    if parsed.path.lower().endswith(
        (".pdf", ".jpg", ".jpeg", ".png", ".zip", ".docx", ".xlsx")
    ):
        return None

    return parsed._replace(
        scheme="https",
        netloc="www.iitdh.ac.in",
        query="",
        fragment=""
    ).geturl().rstrip("/")


def get_words(text):
    return {
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    }


class OfficialWebsiteSearch:
    def __init__(self):
        self.pages = None

    def _crawl(self):
        """Collect text from a limited number of official web pages."""
        queue = deque(SEED_URLS)
        visited = set()
        pages = []

        headers = {
            "User-Agent": "IITDharwadAcademicAssistant/1.0"
        }

        while queue and len(visited) < MAX_PAGES:
            raw_url = queue.popleft()
            url = clean_url(raw_url)

            if not url or url in visited:
                continue

            visited.add(url)

            try:
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=5
                )

                if response.status_code != 200:
                    continue

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if "text/html" not in content_type:
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                for tag in soup([
                    "script", "style", "noscript",
                    "header", "footer", "nav"
                ]):
                    tag.decompose()

                title = soup.title.get_text(" ", strip=True) if soup.title else url
                text = soup.get_text(" ", strip=True)

                if len(text) > 100:
                    pages.append({
                        "title": title,
                        "url": url,
                        "text": text[:8000]
                    })

                links = []

                for link in soup.find_all("a", href=True):
                    next_url = clean_url(urljoin(url, link["href"]))

                    if next_url and next_url not in visited:
                        link_text = link.get_text(" ", strip=True).lower()

                        priority = 0
                        if any(word in (next_url + " " + link_text).lower()
                               for word in (
                                   "academic", "calendar", "announcement",
                                   "student", "hostel", "fee", "exam",
                                   "rule", "registration", "scholarship"
                               )):
                            priority = -1

                        links.append((priority, next_url))

                for _, next_url in sorted(set(links)):
                    if next_url not in visited:
                        queue.append(next_url)

            except requests.RequestException:
                continue

        self.pages = pages

    def search(self, question, top_k=3):
        """Return relevant official website passages with URL citations."""
        if self.pages is None:
            self._crawl()

        query_words = get_words(question)

        if not query_words:
            return []

        results = []

        for page in self.pages:
            page_words = get_words(page["title"] + " " + page["text"])
            matches = query_words.intersection(page_words)

            if not matches:
                continue

            score = len(matches) / float(len(query_words))

            if score >= 0.15:
                results.append({
                    "text": (
                        "Webpage title: " + page["title"] + "\n"
                        + page["text"][:4500]
                    ),
                    "source": page["url"],
                    "page": "webpage",
                    "score": score
                })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]
