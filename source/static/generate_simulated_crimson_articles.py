import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd


load_dotenv()


def main() -> None:
    indir = Path("datastore/output/static/crimson")
    outdir = Path("datastore/output/static/crimson_simulated")
    model = "gpt-4o"
    prompt = (
        "You are rewriting the text of Harvard Crimson opinion articles. "
        "For each article: preserve the argument, facts, and tone of the original article; keep similar "
        "length; rewrite wording and sentence structure. Return JSON only in "
        'this exact shape: {"rewrites":[{"article_id":"...",'
        '"text":"..."}]}. '
        "Input will be JSON in this shape: "
        '{"articles":[{"article_id":"...","text":"..."}]}.'
    )
    n_rows = None
    batch_size = 20
    openai_api_key = os.getenv("OPENAI_API_KEY")

    df_raw = pd.read_csv(indir / "articles_raw.csv")

    df_simulated = generate_simulated_articles(
        df=df_raw,
        model=model,
        prompt=prompt,
        n_rows=n_rows,
        openai_api_key=openai_api_key,
        batch_size=batch_size,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    df_simulated.to_csv(outdir / "articles_simulated.csv", index=False)


def generate_simulated_articles(
    df: pd.DataFrame,
    model: str,
    prompt: str,
    n_rows: int | None,
    openai_api_key: str | None,
    batch_size: int = 20,
) -> pd.DataFrame:
    client = OpenAI(api_key=openai_api_key)

    if n_rows is not None:
        df = df.head(n_rows).copy()
    else:
        df = df.copy()

    n_obs = len(df)
    n_batches = (n_obs + batch_size - 1) // batch_size

    data = []
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_obs)
        print(f"Processing batch {batch_idx + 1}/{n_batches}")
        batch_df = df.iloc[start:end].copy()
        batch_records = (
            batch_df[["article_id", "text"]]
            .assign(
                article_id=lambda x: x["article_id"].astype(str),
                text=lambda x: x["text"].fillna("").astype(str),
            )
            .to_dict(orient="records")
        )
        payload = json.dumps({"articles": batch_records}, ensure_ascii=False)

        response = client.chat.completions.create(
            model=model,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "developer", "content": prompt},
                {"role": "user", "content": payload},
            ],
        )

        content = response.choices[0].message.content
        if content is None:
            continue

        parsed = json.loads(content)
        rewrites = parsed.get("rewrites", [])

        for row in rewrites:
            data.append(
                {
                    "article_id": str(row["article_id"]),
                    "text": row["text"],
                }
            )

    df_rewrites = pd.DataFrame(data)
    if df_rewrites.empty:
        return df

    df_sim = (
        df.assign(article_id=lambda x: x["article_id"].astype(str))
        .merge(df_rewrites, on="article_id", how="left", suffixes=("", "_sim"))
        .assign(text=lambda x: x["text_sim"].combine_first(x["text"]))
        .drop(columns=["text_sim"])
    )

    return df_sim


if __name__ == "__main__":
    main()
