from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    indir = Path("datastore/output/derived/crimson")
    outdir = Path("output/analysis/figure_crimson_ai_generated")
    outdir.mkdir(parents=True, exist_ok=True)

    articles = pd.read_csv(indir / "articles_with_predictions.csv", parse_dates=["date"])

    articles_by_year = (
        articles
        .assign(year=lambda x: x.date.dt.year)
        .assign(has_ai_generated=lambda x: x.share_ai.gt(0))
        .groupby("year", as_index=False)
        .agg(share_ai_generated=("has_ai_generated", "mean"))
        .sort_values(by=["year"])
    )

    articles_by_year.to_csv(outdir / "data_crimson_ai_generated.csv", index=False)
    plot(articles_by_year, outdir / "figure_crimson_ai_generated.png")


def plot(df, path):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="year",
        y="share_ai_generated",
        color="#A51C30",
        ax=ax,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Share AI-generated")
    ax.set_title("AI-Generated articles in the Harvard Crimson")
    ax.set_ylim(0, 0.25)
    ax.tick_params(axis="x", rotation=45)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
