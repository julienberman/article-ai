import hashlib
from pathlib import Path

import pandas as pd


HASH_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def main():
    indir = Path("datastore/output/static/crimson")
    outdir = Path("datastore/output/derived/crimson")

    df = pd.read_csv(
        indir / "articles_raw.csv",
        parse_dates=["date"],
        encoding="utf-8",
    )
    df = clean_articles(df)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "articles_clean.csv", index=False, encoding="utf-8-sig")


def clean_articles(df):
    df_clean = (
        df
        .assign(url=lambda x: x.url.apply(normalize_url))
        .assign(date=lambda x: (
            pd.to_datetime(
                x["url"].str.extract(r"(\d{4}/\d{1,2}/\d{1,2})", expand=False),
                format="%Y/%m/%d",
                errors="coerce",
            )
            .where(lambda s: s.notna(), x["date"])
        ))
        .assign(text=lambda x: (
            x["text"]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        ))
        .loc[lambda x: x.text.notna()]
        .sort_values(by=["date"])
        .drop_duplicates(subset=["url"])
        .assign(article_id=lambda x: x.url.apply(get_article_id))
        [["article_id", "date", "title", "author", "text", "url"]]
    )
    if df_clean.article_id.duplicated().any():
        raise ValueError("Duplicate Crimson article IDs generated")

    return df_clean


def normalize_url(url):
    if str(url).startswith("https://www.thecrimson.com"):
        return str(url)
    return "https://www.thecrimson.com" + str(url)


def get_article_id(url):
    hash_int = int(hashlib.sha1(str(url).encode("utf-8")).hexdigest(), 16)
    url_hash = encode_hash(hash_int, 6)
    return f"crimson_{url_hash}"


def encode_hash(hash_int, length):
    base = len(HASH_ALPHABET)
    hash_int = hash_int % (base ** length)
    chars = []
    for _ in range(length):
        hash_int, remainder = divmod(hash_int, base)
        chars.append(HASH_ALPHABET[remainder])

    return "".join(reversed(chars))


if __name__ == "__main__":
    main()
