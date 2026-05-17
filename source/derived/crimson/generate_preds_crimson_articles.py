import pandas as pd
import pathlib
import os
from pangram import Pangram
from dotenv import load_dotenv

load_dotenv()

def main():
    pangram_api_key = os.getenv("PANGRAM_API_KEY")
    client = Pangram(api_key=pangram_api_key)

    indir = pathlib.Path("datastore/output/derived/crimson")

    if (indir / "articles_with_preds.csv").exists():
        predictions = pd.read_csv(indir / "predictions.csv")
    else:
        predictions = pd.DataFrame()

    articles_clean = pd.read_csv(indir / "articles_clean.csv", parse_dates=["date"], encoding="utf-8")
    articles_clean = (
        articles_clean
        .loc[lambda x: x.date.dt.year >= 2015]
        .groupby(articles_clean.date.dt.year, group_keys=False)
        .sample(n=20, random_state=123)
    )

    unprocessed_articles = get_unprocessed_articles(articles_clean, predictions)

    predictions = pd.concat(predictions, get_predictions(unprocessed_articles, client))

    articles_with_predictions = articles_clean.merge(predictions, how="left", on=["article_id"])

    predictions.to_csv(indir / "predictions.csv", index=False)

    articles_with_predictions.to_csv(indir / "articles_with_predictions.csv", index=False)


def get_unprocessed_articles(articles_clean, predictions):
    return articles_clean.loc[lambda x: ~x.article_id.isin(predictions.article_id)]


def get_predictions(unprocessed_articles, client):
    ids = unprocessed_articles.article_id.to_list()
    texts = unprocessed_articles.text.to_list()
    predictions = []
    for id, text in zip(ids, texts):
        result = client.predict(text, public_dashboard_link=True)
        predictions.append({
            "article_id": id,
            "api_version": result["version"],
            "headline": result["headline"],
            "prediction": result["prediction"],
            "prediction_short": result["prediction_short"],
            "share_ai": result["fraction_ai"],
            "share_ai_assisted": result["fraction_ai_assisted"],
            "share_human": result["fraction_human"],
            "num_ai_segments": result["num_ai_segments"],
            "num_ai_assisted_segments": result["num_ai_assisted_segments"],
            "num_human_segments": result["num_human_segments"],
            "dashboard_link": result["dashboard_link"],
        })

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    main()


