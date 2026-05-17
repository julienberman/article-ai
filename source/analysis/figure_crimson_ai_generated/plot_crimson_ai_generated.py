from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    input_path = Path(
        "datastore/output/derived/crimson/articles_with_predictions.csv"
    )
    output_dir = Path("output/analysis/crimson/figure_crimson_ai_generated")
    output_data_path = output_dir / "figure_crimson_ai_generated.csv"
    output_figure_path = output_dir / "figure_crimson_ai_generated.png"

    output_dir.mkdir(parents=True, exist_ok=True)

    articles = pd.read_csv(input_path, parse_dates=["date"])

    articles_by_year = (
        articles
        .assign(year=lambda x: x.date.dt.year)
        .assign(has_ai_generated=lambda x: x.share_ai.gt(0))
        .groupby("year", as_index=False)
        .agg(share_ai_generated=("has_ai_generated", "mean"))
        .sort_values(by=["year"])
    )

    articles_by_year.to_csv(output_data_path, index=False)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=articles_by_year,
        x="year",
        y="share_ai_generated",
        color="#A51C30",
        ax=ax,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of Articles")
    ax.set_title("Crimson Opinion Articles with AI-Generated Text by Year")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(output_figure_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
