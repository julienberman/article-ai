import pandas as pd
import pathlib
import re


def main():
    indir = pathlib.Path("datastore/output/static/crimson")
    outdir = pathlib.Path("datastore/output/derived/crimson")

    df = pd.read_csv(indir / "articles_raw.csv", parse_dates=["date"], encoding="utf-8")
    df = clean_articles(df)
    df.to_csv(outdir / "articles_clean.csv", index=False, encoding="utf-8-sig")

def clean_articles(df):
    df_clean = (
        df
        .assign(url = lambda x: "https://www.thecrimson.com" + x.url)
        .assign(date=lambda x: pd.to_datetime(
            x["url"].str.extract(r"(\d{4}/\d{2}/\d{2})", expand=False),
            format="%Y/%m/%d",
            errors="coerce",
        ))
        .assign(text=lambda x: x["text"].str.replace(r"\s+", " ", regex=True).str.strip())
        .loc[lambda x: x.text.notna()]
        .sort_values(by=["date"])
        .drop_duplicates()
        .reset_index()
        .assign(article_id = lambda x: "crimson_" + (x.index + 1).astype(str))
        [["article_id", "date", "title", "author", "text", "url"]]
    )
    return df_clean

if __name__ == "__main__":
    main()

