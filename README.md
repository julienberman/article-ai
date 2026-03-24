# Overview

This repository contains code and replication files for ____.


## Project structure

- `source/derived`: data processing code in the SCons pipeline
- `source/analysis`: analysis code in the SCons pipeline
- `source/static`: scraping and model training code outside SCons
- `datastore/raw`: raw datasets
- `datastore/output`: processed datasets
- `output/analysis`: analysis outputs and figures


## Build

- Install dependencies with `uv`
- Run the pipeline with `scons`
- Run only processing with `scons derived`
- Run only analysis with `scons analysis`

Data is available upon request.

