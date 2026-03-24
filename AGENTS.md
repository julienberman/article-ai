# AGENTS.md

## Overview

This project provides code to:
1) Scrape the opinion section of the Harvard Crimson
2) Train a model to predict whether a given text snippet contains AI-driven writing
3) Analyze the Harvard Crimson opinion section and Harvard prize-winning theses over time to see whether content contains AI.
4) Analyze data from the American Time Use Survey on the amount of time that students are spending in class and doing classwork, relative to extracurricular activities

It is a replication package for a guest op-ed that I hope to publish in the New York Times.

## Repository structure

- `source/derived`: Contains code to process raw data. Inside `scons` pipeline.
- `source/analysis`: Contains code to analyze clean data. Inside `scons` pipeline.
- `source/static`: Contains code to scrape the Harvard Crimson and to train supervised models. Outside `scons` pipeline.
- `datastore/raw`: Contains raw data.
- `datastore/output`: Contains clean data.
- `output/analysis`: Contains figures produced in `source/analysis`.

## Build instructions and dependencies

- Dependencies --> Use `uv`
- All Python commands must run inside the project environment via `uv run ...`
- Use `uv sync` before running scripts/tests to ensure dependencies are installed
- Do not use bare `python` or `pip`; use `uv run python ...` and `uv add ...`
- Use `scons` to build --> I want a SConsctruct file in the home directory.
- Env variables should be stored in .env (gitignored)

