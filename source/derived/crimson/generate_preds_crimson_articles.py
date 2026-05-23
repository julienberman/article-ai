import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pangram import Pangram
from tqdm import tqdm

load_dotenv()


def main():
    indir = Path("datastore/output/derived/crimson")
    client = Pangram(api_key=os.getenv("PANGRAM_API_KEY"))

    articles = pd.read_csv(
        indir / "articles_clean.csv",
        parse_dates=["date"],
        encoding="utf-8",
    )
    predictions = load_predictions(indir / "predictions.csv")
    articles_sample = (
        articles
        .loc[lambda x: x.date.dt.year.isin([2024, 2025, 2026])]
    )
    if not predictions.empty:
        articles_sample = articles_sample.loc[
            lambda x: ~x.article_id.isin(predictions.article_id)
        ]

    append_predictions(articles_sample, client, predictions, indir / "predictions.csv")

    predictions = load_predictions(indir / "predictions.csv")
    articles_with_predictions = articles.merge(predictions, how="left", on="article_id")
    articles_with_predictions.to_csv(indir / "articles_with_predictions.csv", index=False)


def load_predictions(predictions_path):
    if not predictions_path.exists():
        return pd.DataFrame()

    df = (
        pd.read_csv(predictions_path)
        .drop_duplicates(subset=["article_id"], keep="last")
    )
    return df


def append_predictions(articles, client, predictions, predictions_path):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(get_prediction, row.article_id, row.text, client)
            for row in articles.itertuples()
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            predictions = (
                pd.concat([predictions, pd.DataFrame([future.result()])])
                .drop_duplicates(subset=["article_id"], keep="last")
            )
            predictions.to_csv(predictions_path, index=False)


def get_prediction(article_id, text, client):
    start = time.perf_counter()
    tqdm.write(f"Beginning {article_id}. | Text: {text[:20]}")
    result = client.predict(text, public_dashboard_link=True)
    time.sleep(1)
    elapsed = time.perf_counter() - start
    tqdm.write(f"Finished {article_id} in {elapsed:.2f} seconds")
    return {
        "article_id": article_id,
        "api_version": result["version"],
        "prediction": result["prediction_short"],
        "share_ai": result["fraction_ai"],
        "share_ai_assisted": result["fraction_ai_assisted"],
        "share_human": result["fraction_human"],
        "num_ai_segments": result["num_ai_segments"],
        "num_ai_assisted_segments": result["num_ai_assisted_segments"],
        "num_human_segments": result["num_human_segments"],
        "dashboard_link": result["dashboard_link"],
    }


if __name__ == "__main__":
    main()
