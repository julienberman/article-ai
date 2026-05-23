import shutil
from pathlib import Path

import pandas as pd

from clean_crimson_articles import clean_articles, get_article_id


def main():
    static_dir = Path("datastore/output/static/crimson")
    derived_dir = Path("datastore/output/derived/crimson")
    raw_path = static_dir / "articles_raw.csv"
    articles_path = derived_dir / "articles_clean.csv"
    predictions_path = derived_dir / "predictions.csv"
    predictions_backup_path = derived_dir / "predictions_pre_hash_migration.csv"

    articles_old = pd.read_csv(articles_path, encoding="utf-8")
    predictions = pd.read_csv(predictions_path, encoding="utf-8")
    id_map = build_id_map(articles_old)

    if not predictions_backup_path.exists():
        shutil.copy2(predictions_path, predictions_backup_path)

    predictions_migrated = migrate_predictions(predictions, id_map)
    predictions_migrated.to_csv(predictions_path, index=False)

    articles_raw = pd.read_csv(
        raw_path,
        parse_dates=["date"],
        encoding="utf-8",
    )
    articles_clean = clean_articles(articles_raw)
    articles_clean.to_csv(articles_path, index=False, encoding="utf-8-sig")

    articles_with_predictions = articles_clean.merge(
        predictions_migrated,
        how="left",
        on="article_id",
    )
    articles_with_predictions.to_csv(
        derived_dir / "articles_with_predictions.csv",
        index=False,
    )


def build_id_map(articles):
    id_map = (
        articles
        .assign(new_article_id=lambda x: x.url.apply(get_article_id))
        .loc[:, ["article_id", "new_article_id"]]
        .drop_duplicates()
    )
    if id_map.article_id.duplicated().any():
        raise ValueError("Existing article IDs are not unique")
    if id_map.new_article_id.duplicated().any():
        raise ValueError("New article IDs are not unique")

    return id_map.set_index("article_id").new_article_id


def migrate_predictions(predictions, id_map):
    missing_ids = set(predictions.article_id).difference(id_map.index)
    if missing_ids:
        raise ValueError(f"Predictions missing article mappings: {missing_ids}")

    predictions_migrated = (
        predictions
        .assign(article_id=lambda x: x.article_id.map(id_map))
        .drop_duplicates(subset=["article_id"], keep="last")
    )
    return predictions_migrated


if __name__ == "__main__":
    main()
