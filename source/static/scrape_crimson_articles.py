import json
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


def main():
    output_dir = Path("datastore/output/static/crimson")
    links_path = output_dir / "article_links.csv"
    articles_path = output_dir / "articles_raw.csv"
    base_url = "https://www.thecrimson.com"
    flush_size = 50
    verbose = True
    tag_urls = {
        # "editorials": "https://www.thecrimson.com/tag/editorials/",
        # "op-eds": "https://www.thecrimson.com/tag/op-eds/",
        "columns": "https://www.thecrimson.com/tag/columns/",
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    article_urls = set()
    for tag_url in tag_urls.values():
        article_urls.update(collect_article_urls(session, tag_url, verbose))

    scraped_urls = load_scraped_urls(articles_path)

    rows_buffer = []
    appended_count = 0
    for article_url in links_df["url"]:
        if article_url in scraped_urls:
            continue

        row = parse_article(session, base_url, article_url, verbose)
        if row:
            row["url"] = article_url
            rows_buffer.append(row)

        if len(rows_buffer) >= flush_size:
            appended_count += flush_rows(rows_buffer, articles_path)
            rows_buffer = []

        time.sleep(0.1)

    appended_count += flush_rows(rows_buffer, articles_path)
    print(f"Appended {appended_count} rows to {articles_path}")


def build_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
    )
    return session


def collect_article_urls(session, tag_url, verbose):
    page_number = 1
    article_urls = []

    while True:
        if page_number == 1:
            page_url = tag_url
        else:
            page_url = f"{tag_url.rstrip('/')}/page/{page_number}/"

        if verbose:
            print(f"GET {page_url}")
        response = session.get(page_url, timeout=30)
        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        page_links = []
        for anchor in soup.select('a[href*="/article/"]'):
            href = anchor.get("href")
            if href and href not in page_links:
                page_links.append(href)

        if not page_links:
            break

        article_urls.extend(page_links)
        page_number += 1
        time.sleep(0.1)

    return article_urls


def parse_article(session, base_url, article_url, verbose):
    full_url = urljoin(base_url, article_url)
    if verbose:
        print(f"GET {full_url}")
    response = session.get(full_url, timeout=30)
    if response.status_code != 200:
        return None

    apollo_state = parse_apollo_state(response.text)
    if not apollo_state:
        return None

    article = find_article_payload(apollo_state)
    if not article:
        return None

    title = article.get("title", "").strip()
    text = extract_text(article)
    if not title or not text:
        return None

    return {
        "author": extract_authors(apollo_state, article),
        "date": article.get("createdOn", ""),
        "title": title,
        "text": text,
    }


def load_scraped_urls(articles_path):
    if not articles_path.exists():
        return set()

    required_columns = {
        "url",
        "author",
        "date",
        "title",
        "text",
    }

    df_header = pd.read_csv(articles_path, nrows=0)
    if not required_columns.issubset(df_header.columns):
        articles_path.unlink()
        return set()

    try:
        df_existing = pd.read_csv(articles_path, usecols=["url"])
    except ValueError:
        return set()

    return set(df_existing["url"].dropna().astype(str))


def flush_rows(rows_buffer, articles_path):
    if not rows_buffer:
        return 0

    columns = ["url", "author", "date", "title", "text"]
    df_buffer = pd.DataFrame(rows_buffer, columns=columns)
    write_header = not articles_path.exists()
    df_buffer.to_csv(
        articles_path,
        mode="a",
        header=write_header,
        index=False,
    )
    return len(df_buffer)


def parse_apollo_state(html):
    marker = "window.__APOLLO_STATE__="
    start = html.find(marker)
    if start == -1:
        return None

    start = start + len(marker)
    first_brace = html.find("{", start)
    if first_brace == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    end = None

    for index in range(first_brace, len(html)):
        char = html[index]
        if in_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            depth += 1
            continue

        if char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end is None:
        return None

    try:
        return json.loads(html[first_brace:end])
    except json.JSONDecodeError:
        return None


def find_article_payload(apollo_state):
    for key, value in apollo_state.items():
        if key.startswith("$ROOT_QUERY.content(") and isinstance(value, dict):
            return value
    return None


def extract_authors(apollo_state, article):
    names = []
    for contributor in article.get("contributors", []):
        contributor_id = contributor.get("id")
        if not contributor_id:
            continue
        contributor_data = apollo_state.get(contributor_id, {})
        name = contributor_data.get("name", "").strip()
        if name and name not in names:
            names.append(name)

    return "; ".join(names)


def extract_text(article):
    paragraphs = article.get("paragraphs", {})
    if not isinstance(paragraphs, dict):
        return ""

    text_parts = []
    for paragraph_html in paragraphs.get("json", []):
        paragraph_text = (
            BeautifulSoup(
                paragraph_html,
                "html.parser",
            )
            .get_text(" ")
            .strip()
        )
        if not paragraph_text:
            continue
        if paragraph_text.startswith("{") and paragraph_text.endswith("}"):
            continue
        text_parts.append(paragraph_text)

    return " ".join(text_parts)


if __name__ == "__main__":
    main()
