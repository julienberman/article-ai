from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    indir = Path("datastore/output/derived/crimson")
    outdir = Path("output/analysis/figure_crimson_totals")
    outdir.mkdir(parents=True, exist_ok=True)

    articles = pd.read_csv(indir / "articles_clean.csv", parse_dates=["date"])

    articles_by_year = (
        articles
        .assign(year=lambda x: x.date.dt.year)
        .loc[lambda x: x.year >= 2000]
        .groupby("year", as_index=False)
        .agg(n_articles=("article_id", "count"))
        .assign(year = lambda x: x.year.astype(int))
        .sort_values(by=["year"])
    )

    articles_by_year.to_csv(outdir / "data_crimson_totals.csv", index=False)

    plot(articles_by_year, outdir / "figure_crimson_total.png")
    
def plot(df, output_figure_path):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="year",
        y="n_articles",
        color="#A51C30",
        ax=ax,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Opinion articles published in the Harvard Crimson")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(output_figure_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
