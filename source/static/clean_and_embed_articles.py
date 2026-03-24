from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path

load_dotenv()


def main():
    INDIR = Path("datastore/output/static/crimson")
    MODEL = "text-embedding-3-large"
    df_real = pd.read_csv(INDIR / "articles_raw.csv")
    df_sim = pd.read_csv(INDIR / "articles_sim.csv")
    df = clean(df_real, df_sim)
    df = embed(df, model=MODEL, batch_size=64)
    df.to_parquet(INDIR / "articles_with_embeddings.parquet")


def clean(df_real, df_sim):
    df_real = df_real.assign(is_sim=0)
    df_sim = df_sim.assign(is_sim=1)
    df = pd.concat([df_real, df_sim])
    df_clean = (
        df.assign(id=lambda x: range(1, len(x) + 1))
        .assign(text=lambda x: x["text"].str.lower())
        .assign(date=lambda x: x["date"])
    )
    return df_clean


def embed(df, model="text-embedding-3-large", batch_size=64):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    df = df.reset_index(drop=True)
    n_obs = len(df)
    n_batches = (n_obs + batch_size - 1) // batch_size
    data = []

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_obs)
        print(f"Processing batch {batch_idx + 1}/{n_batches}")

        batch_df = df.iloc[start:end]
        batch_ids = batch_df["id"].tolist()
        batch_texts = batch_df["text"].tolist()

        response = client.embeddings.create(
            model=model,
            input=batch_texts,
        )

        for id, emb_obj in zip(batch_ids, response.data):
            data.append(
                {
                    "id": id,
                    "embedding": emb_obj.embedding,
                }
            )

    df_embeddings = pd.DataFrame(data)
    df_with_embeddings = df.merge(df_embeddings, on="id")
    return df_with_embeddings


if __name__ == "__main__":
    main()
