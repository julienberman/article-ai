import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def main() -> None:
    args = parse_args()
    indir = Path("datastore/output/static/crimson")
    outdir = Path("datastore/output/static/crimson")
    prompt = build_prompt()
    run_start = time.perf_counter()

    outdir.mkdir(parents=True, exist_ok=True)
    batches_dir = outdir / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "create":
        df_raw = pd.read_csv(indir / "articles_raw.csv").drop_duplicates()
        create_batch_files(
            df=df_raw,
            outdir=outdir,
            batches_dir=batches_dir,
            model=args.model,
            prompt=prompt,
            n_rows=args.n_rows,
        )

    if args.mode == "submit":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        submit_batch_jobs(
            client=client,
            outdir=outdir,
            max_submit_retries=args.max_submit_retries,
            submit_backoff_seconds=args.submit_backoff_seconds,
            submit_max_backoff_seconds=args.submit_max_backoff_seconds,
        )

    if args.mode == "status":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        refresh_batch_statuses(client=client, outdir=outdir)

    if args.mode == "process":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        process_batch_outputs(client=client, outdir=outdir)

    total_seconds = time.perf_counter() - run_start
    run_timing = {
        "mode": args.mode,
        "total_seconds": total_seconds,
    }
    with (outdir / "run_timing.json").open("w", encoding="utf-8") as handle:
        json.dump(run_timing, handle, ensure_ascii=False, indent=2)

    print(f"Mode {args.mode} completed in {total_seconds:.2f} seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["create", "submit", "status", "process"],
        required=True,
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--n-rows", type=int, default=None)
    parser.add_argument("--max-submit-retries", type=int, default=8)
    parser.add_argument("--submit-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--submit-max-backoff-seconds", type=float, default=120.0)
    return parser.parse_args()


def build_prompt() -> str:
    return (
        "You are rewriting the text of Harvard Crimson opinion articles. "
        "For this article: preserve the argument, facts, and tone of the "
        "original article; keep similar length; rewrite wording and sentence "
        "structure. Return JSON only in this exact shape: "
        '{"article_id":"...","text":"..."}. '
        "Input will be JSON in this shape: "
        '{"article_id":"...","text":"..."}.'
    )


def generate_article_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        article_id=lambda x: (
            x["url"].fillna("").astype(str) + "|" + x["date"].fillna("").astype(str)
        ).map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:8])
    )


def create_batch_files(
    df: pd.DataFrame,
    outdir: Path,
    batches_dir: Path,
    model: str,
    prompt: str,
    n_rows: int | None,
) -> None:
    if n_rows is not None:
        df = df.head(n_rows).copy()
    else:
        df = df.copy()

    df = generate_article_ids(df)
    if df["article_id"].duplicated().any():
        raise ValueError("article_id hash collisions detected")

    source_cols = [
        "article_id",
        "url",
        "author",
        "date",
        "title",
        "text",
    ]
    df_source = (
        df.assign(article_id=lambda x: x["article_id"].astype(str))
        .loc[:, source_cols]
        .reset_index(drop=True)
    )

    clear_existing_batch_files(batches_dir=batches_dir)

    batches = split_into_batches(
        df=df_source,
        prompt=prompt,
    )

    data = []
    for batch_idx, batch_df in enumerate(batches):
        batch_num = batch_idx + 1
        batch_path = batches_dir / f"batch_{batch_num:04d}.jsonl"
        estimated_tokens = write_batch_file(
            batch_df=batch_df,
            batch_path=batch_path,
            model=model,
            prompt=prompt,
        )

        data.append(
            {
                "batch_number": batch_num,
                "batch_file": str(batch_path),
                "n_requests": int(len(batch_df)),
                "start_row": int(batch_df["row_index"].min()),
                "end_row": int(batch_df["row_index"].max()) + 1,
                "estimated_tokens": int(estimated_tokens),
            }
        )

    batch_files = pd.DataFrame(data)
    batch_files.to_csv(outdir / "batch_files_manifest.csv", index=False)
    print(f"Created {len(batch_files)} batch files in {batches_dir}")


def clear_existing_batch_files(batches_dir: Path) -> None:
    for file_path in batches_dir.glob("batch_*.jsonl"):
        file_path.unlink()


def split_into_batches(
    df: pd.DataFrame,
    prompt: str,
) -> list[pd.DataFrame]:
    batches = []
    current_rows = []

    for row_index, row in df.iterrows():
        article_id = str(row["article_id"])
        text = "" if pd.isna(row["text"]) else str(row["text"])
        row_tokens = estimate_request_tokens(
            article_id=article_id,
            text=text,
            prompt=prompt,
        )

        if len(current_rows) >= 64:
            batch_df = pd.DataFrame(current_rows)
            batches.append(batch_df)
            current_rows = []

        current_rows.append(
            {
                "row_index": int(row_index),
                "article_id": article_id,
                "url": str(row["url"]),
                "author": str(row["author"]),
                "date": str(row["date"]),
                "title": str(row["title"]),
                "text": text,
                "estimated_request_tokens": int(row_tokens),
            }
        )

    if current_rows:
        batch_df = pd.DataFrame(current_rows)
        batches.append(batch_df)

    return batches


def write_batch_file(
    batch_df: pd.DataFrame,
    batch_path: Path,
    model: str,
    prompt: str,
) -> int:
    total_tokens = int(batch_df["estimated_request_tokens"].sum())

    with batch_path.open("w", encoding="utf-8") as handle:
        records = batch_df.to_dict(orient="records")
        for row in records:
            payload = json.dumps(
                {
                    "article_id": str(row["article_id"]),
                    "text": str(row["text"]),
                },
                ensure_ascii=False,
            )
            request = {
                "custom_id": str(row["article_id"]),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "developer", "content": prompt},
                        {"role": "user", "content": payload},
                    ],
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    return total_tokens


def estimate_request_tokens(article_id: str, text: str, prompt: str) -> int:
    payload = json.dumps(
        {
            "article_id": article_id,
            "text": text,
        },
        ensure_ascii=False,
    )

    input_tokens = estimate_tokens(prompt) + estimate_tokens(payload) + 200
    output_tokens = int(estimate_tokens(text) * 1.35) + 200
    return input_tokens + output_tokens


def estimate_tokens(text: str) -> int:
    if text == "":
        return 1
    return int(math.ceil(len(text) / 4))


def submit_batch_jobs(
    client: OpenAI,
    outdir: Path,
    max_submit_retries: int,
    submit_backoff_seconds: float,
    submit_max_backoff_seconds: float,
) -> None:
    batch_files_path = outdir / "batch_files_manifest.csv"
    jobs_path = outdir / "batch_jobs_manifest.csv"

    if not batch_files_path.exists():
        raise FileNotFoundError("Run --mode create before --mode submit")

    batch_files = pd.read_csv(batch_files_path)
    batch_jobs = load_batch_jobs(jobs_path=jobs_path)
    if "estimated_tokens" in batch_jobs.columns:
        missing_tokens = batch_jobs["estimated_tokens"].isna()
        if missing_tokens.any():
            token_lookup = batch_files.loc[
                :, ["batch_file", "estimated_tokens"]
            ].assign(batch_file=lambda x: x["batch_file"].astype(str))
            batch_jobs = (
                batch_jobs.assign(batch_file=lambda x: x["batch_file"].astype(str))
                .merge(
                    token_lookup,
                    how="left",
                    on="batch_file",
                    suffixes=("", "_from_files"),
                )
                .assign(
                    estimated_tokens=lambda x: x["estimated_tokens"].combine_first(
                        x["estimated_tokens_from_files"]
                    )
                )
                .drop(columns=["estimated_tokens_from_files"])
            )
    batch_jobs = refresh_batch_jobs_dataframe(client=client, batch_jobs=batch_jobs)

    submitted_files = list(batch_jobs["batch_file"].dropna().astype(str))
    pending = batch_files.loc[
        ~batch_files["batch_file"].astype(str).isin(submitted_files)
    ].sort_values(by=["batch_number"])

    new_rows = []
    for _, row in pending.iterrows():
        batch_number = int(row["batch_number"])
        batch_file = str(row["batch_file"])
        estimated_tokens = int(row["estimated_tokens"])

        uploaded = upload_batch_file_with_backoff(
            client=client,
            batch_file=Path(batch_file),
            max_submit_retries=max_submit_retries,
            submit_backoff_seconds=submit_backoff_seconds,
            submit_max_backoff_seconds=submit_max_backoff_seconds,
            batch_number=batch_number,
        )

        batch_job = create_batch_job_with_backoff(
            client=client,
            input_file_id=str(uploaded.id),
            max_submit_retries=max_submit_retries,
            submit_backoff_seconds=submit_backoff_seconds,
            submit_max_backoff_seconds=submit_max_backoff_seconds,
            batch_number=batch_number,
        )

        print(
            f"Submitted batch {batch_number}: {batch_job.id} status={batch_job.status}"
        )

        new_rows.append(
            {
                "batch_number": batch_number,
                "batch_file": batch_file,
                "n_requests": int(row["n_requests"]),
                "estimated_tokens": estimated_tokens,
                "input_file_id": uploaded.id,
                "batch_id": batch_job.id,
                "status": str(batch_job.status),
                "output_file_id": to_optional_str(batch_job.output_file_id),
                "error_file_id": to_optional_str(batch_job.error_file_id),
            }
        )

    if new_rows:
        batch_jobs = pd.concat([batch_jobs, pd.DataFrame(new_rows)], ignore_index=True)

    batch_jobs.to_csv(jobs_path, index=False)
    print(f"Saved job manifest to {jobs_path}")


def upload_batch_file_with_backoff(
    client: OpenAI,
    batch_file: Path,
    max_submit_retries: int,
    submit_backoff_seconds: float,
    submit_max_backoff_seconds: float,
    batch_number: int,
) -> object:
    attempt = 0
    while True:
        try:
            with batch_file.open("rb") as handle:
                return client.files.create(file=handle, purpose="batch")
        except Exception as error:
            if (
                should_retry_submission_error(error=error)
                and attempt < max_submit_retries
            ):
                wait_seconds = compute_backoff_delay(
                    attempt=attempt,
                    submit_backoff_seconds=submit_backoff_seconds,
                    submit_max_backoff_seconds=submit_max_backoff_seconds,
                )
                print(
                    "Retrying upload for batch "
                    f"{batch_number} in {wait_seconds:.1f}s after error: {error}"
                )
                time.sleep(wait_seconds)
                attempt += 1
                continue
            raise


def create_batch_job_with_backoff(
    client: OpenAI,
    input_file_id: str,
    max_submit_retries: int,
    submit_backoff_seconds: float,
    submit_max_backoff_seconds: float,
    batch_number: int,
) -> object:
    attempt = 0
    while True:
        try:
            return client.batches.create(
                input_file_id=input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
        except Exception as error:
            if (
                should_retry_submission_error(error=error)
                and attempt < max_submit_retries
            ):
                wait_seconds = compute_backoff_delay(
                    attempt=attempt,
                    submit_backoff_seconds=submit_backoff_seconds,
                    submit_max_backoff_seconds=submit_max_backoff_seconds,
                )
                print(
                    "Retrying batch creation for batch "
                    f"{batch_number} in {wait_seconds:.1f}s after error: {error}"
                )
                time.sleep(wait_seconds)
                attempt += 1
                continue
            raise


def should_retry_submission_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {408, 409, 429, 500, 502, 503, 504}:
            return True

    error_text = str(error).lower()
    retry_markers = [
        "rate limit",
        "too many requests",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "try again",
        "enqueued",
    ]
    return any(marker in error_text for marker in retry_markers)


def compute_backoff_delay(
    attempt: int,
    submit_backoff_seconds: float,
    submit_max_backoff_seconds: float,
) -> float:
    raw_delay = submit_backoff_seconds * (2**attempt)
    clipped_delay = min(raw_delay, submit_max_backoff_seconds)
    jittered_delay = clipped_delay * random.uniform(0.8, 1.2)
    return max(jittered_delay, 0.0)


def load_batch_jobs(jobs_path: Path) -> pd.DataFrame:
    columns = [
        "batch_number",
        "batch_file",
        "n_requests",
        "estimated_tokens",
        "input_file_id",
        "batch_id",
        "status",
        "output_file_id",
        "error_file_id",
    ]
    if not jobs_path.exists():
        return pd.DataFrame(columns=columns)
    batch_jobs = pd.read_csv(jobs_path)
    for column in columns:
        if column not in batch_jobs.columns:
            batch_jobs[column] = pd.NA
    return batch_jobs.loc[:, columns]


def refresh_batch_statuses(client: OpenAI, outdir: Path) -> None:
    jobs_path = outdir / "batch_jobs_manifest.csv"
    batch_jobs = load_batch_jobs(jobs_path=jobs_path)
    if batch_jobs.empty:
        raise FileNotFoundError("No batch jobs found. Run --mode submit first")

    batch_jobs = refresh_batch_jobs_dataframe(client=client, batch_jobs=batch_jobs)
    batch_jobs.to_csv(jobs_path, index=False)
    print(f"Updated statuses in {jobs_path}")


def refresh_batch_jobs_dataframe(
    client: OpenAI,
    batch_jobs: pd.DataFrame,
) -> pd.DataFrame:
    if batch_jobs.empty:
        return batch_jobs

    refreshed = batch_jobs.copy()
    for idx, row in refreshed.iterrows():
        batch_id = to_optional_str(row.get("batch_id"))
        if batch_id is None:
            continue

        batch_job = client.batches.retrieve(batch_id)
        refreshed.at[idx, "status"] = str(batch_job.status)
        refreshed.at[idx, "output_file_id"] = to_optional_str(batch_job.output_file_id)
        refreshed.at[idx, "error_file_id"] = to_optional_str(batch_job.error_file_id)

    return refreshed


def process_batch_outputs(client: OpenAI, outdir: Path) -> None:
    source_path = outdir / "articles_raw.csv"
    jobs_path = outdir / "batch_jobs_manifest.csv"

    if not source_path.exists():
        raise FileNotFoundError("Missing articles_raw.csv")
    if not jobs_path.exists():
        raise FileNotFoundError("Missing batch_jobs_manifest.csv")

    df_source = (
        pd.read_csv(source_path)
        .pipe(generate_article_ids)
        .assign(article_id=lambda x: x["article_id"].astype(str))
        .loc[:, ["article_id", "url", "author", "date", "title", "text"]]
    )
    batch_jobs = load_batch_jobs(jobs_path=jobs_path)
    batch_jobs = refresh_batch_jobs_dataframe(client=client, batch_jobs=batch_jobs)
    batch_jobs.to_csv(jobs_path, index=False)

    requested_ids = set()
    for _, row in batch_jobs.iterrows():
        batch_file = Path(str(row["batch_file"]))
        if not batch_file.exists():
            continue
        request_ids = read_batch_request_ids(batch_file)
        requested_ids.update(request_ids)

    if requested_ids:
        df_source = df_source.loc[
            df_source["article_id"].isin(list(requested_ids))
        ].copy()

    diagnostics = (
        df_source.loc[:, ["article_id", "url", "author", "date", "title"]]
        .assign(
            simulation_status="missing",
            error_message="",
            batch_number=pd.NA,
            batch_id=pd.NA,
        )
        .set_index("article_id")
    )

    rewrites = {}
    for _, row in batch_jobs.iterrows():
        batch_number = int(row["batch_number"])
        batch_id = to_optional_str(row["batch_id"])
        status = str(row["status"])
        output_file_id = to_optional_str(row["output_file_id"])
        error_file_id = to_optional_str(row["error_file_id"])
        request_ids = read_batch_request_ids(Path(str(row["batch_file"])))

        for request_id in request_ids:
            if request_id not in diagnostics.index:
                continue
            diagnostics.at[request_id, "batch_number"] = batch_number
            diagnostics.at[request_id, "batch_id"] = batch_id
            if status in {"failed", "cancelled", "expired"}:
                diagnostics.at[request_id, "simulation_status"] = "failed"
                diagnostics.at[request_id, "error_message"] = f"batch_status={status}"

        if status == "completed" and output_file_id is not None:
            apply_output_file(
                client=client,
                output_file_id=output_file_id,
                diagnostics=diagnostics,
                rewrites=rewrites,
            )

        if error_file_id is not None:
            apply_error_file(
                client=client,
                error_file_id=error_file_id,
                diagnostics=diagnostics,
            )

    df_sim = df_source.assign(text=lambda x: x["article_id"].map(rewrites)).loc[
        lambda x: x["text"].notna(),
        [
            "article_id",
            "url",
            "author",
            "date",
            "title",
            "text",
        ],
    ]
    df_sim.to_csv(outdir / "articles_sim.csv", index=False)

    diagnostics = diagnostics.reset_index()
    diagnostics.to_csv(outdir / "simulation_diagnostics.csv", index=False)
    print(f"Wrote {len(df_sim)} rows to {outdir / 'articles_sim.csv'}")


def read_batch_request_ids(batch_file: Path) -> list[str]:
    ids = []
    with batch_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip() == "":
                continue
            record = json.loads(raw_line)
            custom_id = str(record.get("custom_id", "")).strip()
            if custom_id != "":
                ids.append(custom_id)
    return ids


def apply_output_file(
    client: OpenAI,
    output_file_id: str,
    diagnostics: pd.DataFrame,
    rewrites: dict[str, str],
) -> None:
    output_text = read_batch_file_content(client=client, file_id=output_file_id)
    for raw_line in output_text.splitlines():
        if raw_line.strip() == "":
            continue

        record = json.loads(raw_line)
        custom_id = str(record.get("custom_id", "")).strip()
        if custom_id == "" or custom_id not in diagnostics.index:
            continue

        response = record.get("response")
        if not isinstance(response, dict):
            diagnostics.at[custom_id, "simulation_status"] = "failed"
            diagnostics.at[custom_id, "error_message"] = "missing response"
            continue

        status_code = int(response.get("status_code", 0))
        if status_code != 200:
            diagnostics.at[custom_id, "simulation_status"] = "failed"
            diagnostics.at[custom_id, "error_message"] = f"status_code={status_code}"
            continue

        body = response.get("body", {})
        content = extract_message_content(body)
        if content is None:
            diagnostics.at[custom_id, "simulation_status"] = "failed"
            diagnostics.at[custom_id, "error_message"] = "empty content"
            continue

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            diagnostics.at[custom_id, "simulation_status"] = "failed"
            diagnostics.at[custom_id, "error_message"] = "invalid json output"
            continue

        parsed_article_id = str(parsed.get("article_id", custom_id)).strip()
        simulated_text = str(parsed.get("text", "")).strip()
        if simulated_text == "":
            diagnostics.at[custom_id, "simulation_status"] = "failed"
            diagnostics.at[custom_id, "error_message"] = "empty text"
            continue

        rewrites[parsed_article_id] = simulated_text
        diagnostics.at[custom_id, "simulation_status"] = "success"
        diagnostics.at[custom_id, "error_message"] = ""


def apply_error_file(
    client: OpenAI,
    error_file_id: str,
    diagnostics: pd.DataFrame,
) -> None:
    error_text = read_batch_file_content(client=client, file_id=error_file_id)
    for raw_line in error_text.splitlines():
        if raw_line.strip() == "":
            continue

        record = json.loads(raw_line)
        custom_id = str(record.get("custom_id", "")).strip()
        if custom_id == "" or custom_id not in diagnostics.index:
            continue

        diagnostics.at[custom_id, "simulation_status"] = "failed"
        diagnostics.at[custom_id, "error_message"] = str(
            record.get("error", "batch error")
        )


def read_batch_file_content(client: OpenAI, file_id: str) -> str:
    content = client.files.content(file_id)
    if hasattr(content, "text") and isinstance(content.text, str):
        return content.text
    if hasattr(content, "content"):
        raw = content.content
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
    return str(content)


def extract_message_content(body: dict) -> str | None:
    choices = body.get("choices", [])
    if not choices:
        return None

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)

    return None


def to_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


if __name__ == "__main__":
    main()
