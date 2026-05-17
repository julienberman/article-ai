import pandas as pd


def main():
    indir_articles_clean = pathlib.Path("datastore/output/derived/crimson")
    indir_articles_with_preds = pathlib.Path("datastore/output/static/crimson")
    if (indir_articles_with_preds / "articles_with_preds.csv").exists():
        articles_with_preds = pd.read_csv(indir_articles_with_preds / "articles_with_preds.csv")
    else:
        articles_with_preds = pd.DataFrame()

    articles_clean = pd.read_csv(indir_articles_clean / "articles_clean.csv", parse_dates=["date"], encoding="utf-8")

    unprocessed_articles = get_unprocessed_articles(articles_clean, articles_with_preds)

    processed_articles = process_articles(unprocessed_articles)

    articles_with_preds = (
        pd.concat(articles_with_preds, processed_articles)
        .sort_values(by=article_id)
    )

    articles_with_preds.to_csv(indir_articles_with_preds / "articles_with_preds.csv", index=False)


def get_unprocessed_articles(articles_clean, articles_with_preds):
    return articles_clean.loc[lambda x: ~x.article_id.isin(articles_with_preds.article_id)]

def process_articles(unprocessed_articles):

    pass

if __name__ == "__main__":
    main()


