from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    input_path = Path(
        "datastore/output/derived/crimson/articles_with_predictions.csv"
    )
    output_dir = Path("output/analysis/crimson/figure_crimson_totals")
    output_data_path = output_dir / "figure_crimson_totals.csv"
    output_figure_path = output_dir / "figure_crimson_totals.png"

    output_dir.mkdir(parents=True, exist_ok=True)

    articles = pd.read_csv(input_path, parse_dates=["date"])

    articles_by_year = (
        articles
        .assign(year=lambda x: x.date.dt.year)
        .groupby("year", as_index=False)
        .agg(n_articles=("article_id", "count"))
        .sort_values(by=["year"])
    )

    articles_by_year.to_csv(output_data_path, index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=articles_by_year,
        x="year",
        y="n_articles",
        color="#A51C30",
        ax=ax,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Crimson Opinion Articles Published by Year")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(output_figure_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
