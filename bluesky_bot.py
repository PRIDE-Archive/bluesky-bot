import os
from fastapi import FastAPI
from blueskysocial import Client, Post
import logging
import uvicorn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import click
import configparser

app = FastAPI(
    title="bluesky pride bot",
    description="bluesky bot - publish dataset for PRIDE Archive",
    version="0.0.1",
    contact={
        "name": "PRIDE Team",
        "url": "https://www.ebi.ac.uk/pride/",
        "email": "pride-support@ebi.ac.uk",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Set up the Bluesky client
BLUESKY_HANDLE = None  # os.getenv("BLUESKY_HANDLE")  # Your Bluesky handle (e.g., user.bsky.social)
BLUESKY_PASSWORD = None  # os.getenv("BLUESKY_PASSWORD")  # Your Bluesky password

client = Client()

# Load FLAN-T5 small model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Define the summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


def build_bluesky_post(accession, tweet, url) -> str:
    """
    Build a post for Bluesky.
    :param accession: Accession
    :param title: Title
    :param description: description
    :param url: URL to the dataset
    :return:
    """
    # Define the alert emoji
    alert_emoji = "🚨"

    # Construct the final post with emoji, accession, and URL
    final_post = f"{alert_emoji} {accession} {alert_emoji}\n\n{tweet}\n\n {url}"

    return final_post

def create_tweet(title, description):
    prompt = f"Summarize this dataset for social media: Title: {title}; Description: {description}"

    # Generate a summary with a character limit
    result = summarizer(prompt, max_length=60, min_length=10, do_sample=True)[0]['summary_text']

    return result[:239]


@app.get("/publish")
async def post_to_bluesky(accession: str, title: str, description: str, url: str):
    tweet = create_tweet(title, description)
    post_str = build_bluesky_post(accession, tweet, url)
    post = Post(post_str)
    client.post(post)


class NoHealthAccessLogFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if "GET /health" in message:
            return False
        else:
            return True


def get_config(file):
    """
    This method read the default configuration file config.ini in the same path of the pipeline execution
    :return:
    """
    config = configparser.ConfigParser()
    config.read(file)
    return config


@click.command()
@click.option("--config-file", "-a", type=click.Path(), default="config.ini")
@click.option(
    "--config-profile",
    "-c",
    help="This option allow to select a config profile",
    default="TEST",
)
def main(config_file, config_profile):
    global BLUESKY_HANDLE, BLUESKY_PASSWORD
    config = get_config(config_file)
    BLUESKY_HANDLE = config[config_profile]["BLUESKY_HANDLE"]
    BLUESKY_PASSWORD = config[config_profile]["BLUESKY_PASSWORD"]
    PORT = config[config_profile]["PORT"]

    client.authenticate(BLUESKY_HANDLE, BLUESKY_PASSWORD)

    uvicorn.run(app, host="0.0.0.0", port=int(PORT))


@app.get("/health")
def read_docs():
    return "alive"

if __name__ == "__main__":
    main()

