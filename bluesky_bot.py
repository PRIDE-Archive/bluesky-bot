import configparser
import logging

import click
from blueskysocial import Client, Post
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import random

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
buffer = []

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
    final_post = f"[{accession}]({url}) {alert_emoji}\n\n{tweet}\n\n{alert_emoji} New dataset alert! {alert_emoji}"

    return final_post

def create_tweet(title, description):
    prompt = f"Summarize this dataset for social media with 200 characters: Title: {title}; Description: {description}"

    # Generate a summary with a character limit
    result = summarizer(prompt, max_length=60, min_length=10, do_sample=True)[0]['summary_text']

    return result[:239]

class MessageModel(BaseModel):
    accession: str
    title: str
    description: str
    url: str


@app.post("/publish")
async def post_to_bluesky(message: MessageModel):
    tweet = create_tweet(message.title, message.description)
    post_str = build_bluesky_post(message.accession, tweet, message.url)
    # post = Post(post_str)
    # client.post(post)
    buffer.append(post_str)
    return {"status": "Added to buffer", "total_in_buffer": len(buffer)}

@app.get("/buffer_count")
async def get_buffer_count():
    """
    Get the number of posts currently in the buffer.
    :return: Number of posts in the buffer.
    """
    count = len(buffer)
    return {"pending_posts": count}


def fix_post_urls(post_content, urls):
    """
    Fix the URLs in the post content.
    :param post_content: The post content.
    :param urls: The URLs to be fixed.
    :return: The post content with fixed URLs.
    """
    post_content._post['text'] = post_content._post['text'].replace("[", "").replace("]", "").replace("(h","")
    return post_content


def post_from_buffer():
    if buffer:
        post_content = random.choice(buffer)  # Select a random post from the buffer
        buffer.remove(post_content)  # Remove the selected post from the buffer
        post = Post(post_content)
        urls = post._parse_rich_urls()
        logging.info(f"Posting: {urls}")
        client.post(post)
        print(f"Posted at {datetime.now().strftime('%H:%M')} - {post_content}")
    else:
        print(f"No posts in buffer to post at {datetime.now().strftime('%H:%M')}")

@app.get("/post_now")
async def post_now():
    post_from_buffer()
    return {"status": "Posted"}

def clear_buffer():
    global buffer
    buffer = []
    print("Buffer cleared for the new day.")

# Initialize the scheduler
scheduler = BackgroundScheduler()

# Schedule posting times
scheduler.add_job(post_from_buffer, CronTrigger(hour=7, minute=0))
scheduler.add_job(post_from_buffer, CronTrigger(hour=9, minute=0))
scheduler.add_job(post_from_buffer, CronTrigger(hour=11, minute=0))
scheduler.add_job(post_from_buffer, CronTrigger(hour=14, minute=0))
scheduler.add_job(post_from_buffer, CronTrigger(hour=16, minute=0))
scheduler.add_job(post_from_buffer, CronTrigger(hour=18, minute=0))
scheduler.add_job(post_from_buffer, CronTrigger(hour=22, minute=0))

# Schedule buffer clearing at midnight
scheduler.add_job(clear_buffer, CronTrigger(hour=0, minute=0))

# Start the scheduler
scheduler.start()


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

    logging.getLogger("uvicorn.access").addFilter(NoHealthAccessLogFilter())

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(PORT))


@app.get("/health")
def read_docs():
    return "alive"

if __name__ == "__main__":
    main()

