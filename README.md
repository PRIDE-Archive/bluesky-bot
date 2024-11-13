# Bluesky PRIDE Bot

This project is a FastAPI-based bot designed to publish dataset updates from the PRIDE Archive to Bluesky. The bot uses the FLAN-T5 model to summarize dataset titles and descriptions and posts them to Bluesky with an alert emoji for easy visibility.

## Features

- **Summarization**: The bot uses a pre-trained FLAN-T5 small model to summarize dataset titles and descriptions for social media.
- **Bluesky Integration**: The bot posts the dataset information to Bluesky using the BlueskySocial Python client.
- **FastAPI Endpoint**: Exposes an endpoint (`/publish`) to trigger posts to Bluesky.
- **Customizable**: Easily configurable with your Bluesky credentials and dataset information.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- transformers
- blueskysocial
- torch

You can install the necessary dependencies using:

```bash
pip install fastapi uvicorn transformers blueskysocial torch
