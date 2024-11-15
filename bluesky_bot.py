import configparser
import logging
import os
import random
import asyncio
import re
from datetime import datetime
from typing import List, Dict

import click
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import httpx
from aiolimiter import AsyncLimiter

# Logging setup
logging.basicConfig(level=logging.INFO)

# Constants
BUFFER = []
EXISTING_POSTS_BUFFER = []

# FastAPI setup
app = FastAPI(
    title="Bluesky PRIDE Bot",
    description="Bluesky bot - publish datasets for PRIDE Archive",
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

# Configuration placeholders
BLUESKY_HANDLE = None
BLUESKY_PASSWORD = None

# Load FLAN-T5 model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Rate limiters
five_minute_limiter = AsyncLimiter(30, 300)
daily_limiter = AsyncLimiter(300, 86400)


async def bluesky_login():
    """Log in to Bluesky and return the JWT token."""
    async with five_minute_limiter, daily_limiter:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://bsky.social/xrpc/com.atproto.server.createSession",
                json={"identifier": BLUESKY_HANDLE, "password": BLUESKY_PASSWORD},
            )
            response.raise_for_status()
            return response.json()["accessJwt"]


def build_bluesky_post(accession: str, tweet: str, url: str) -> str:
    """Build a Bluesky post."""
    alert_emoji = "🚨"
    return f"[{accession}]({url}) {alert_emoji}\n\n{tweet}\n\n{alert_emoji} New dataset alert! {alert_emoji}"


def create_tweet(title: str, description: str) -> str:
    """Create a tweet using LLM to summarize the dataset."""
    prompt = f"Summarize this dataset for social media with 200 characters: Title: {title}; Description: {description}"
    result = summarizer(prompt, max_length=60, min_length=10, do_sample=True)[0]["summary_text"]
    return result[:239]


class MessageModel(BaseModel):
    accession: str
    title: str
    description: str
    url: str


@app.post("/publish")
async def post_to_bluesky(message: MessageModel):
    """Add a message to the posting buffer."""
    tweet = create_tweet(message.title, message.description)
    post_str = build_bluesky_post(message.accession, tweet, message.url)
    BUFFER.append(post_str)
    return {"status": "Added to buffer", "total_in_buffer": len(BUFFER)}


@app.get("/buffer_count")
async def get_buffer_count():
    """Get the number of posts in the buffer."""
    return {"pending_posts": len(BUFFER)}


@app.get("/post_now")
async def post_now():
    """Manually post from the buffer."""
    if BUFFER:
        await post_from_buffer()
        return {"status": "Posted"}
    return {"status": "Buffer is empty"}


@app.get("/get_posts")
async def get_posts(limit: int = 5):
    """Fetch recent posts from Bluesky."""
    try:
        jwt_token = await bluesky_login()
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {jwt_token}"}
            response = await client.get(
                "https://bsky.social/xrpc/com.atproto.repo.listRecords",
                headers=headers,
                params={"repo": BLUESKY_HANDLE, "collection": "app.bsky.feed.post", "limit": limit},
            )
            response.raise_for_status()
            records = response.json().get("records", [])
            global EXISTING_POSTS_BUFFER
            EXISTING_POSTS_BUFFER = [
                {
                    "id": record["uri"].split("/")[-1],
                    "content": record["value"]["text"],
                    "createdAt": record["value"]["createdAt"],
                }
                for record in records
            ]
    except Exception as e:
        logging.error(f"Error fetching posts: {e}")
    return EXISTING_POSTS_BUFFER

def _parse_urls(post_text: str) -> List[Dict]:
    """
    Parse plain URLs from the post text.

    Args:
        post_text (str): The text of the post.

    Returns:
        List[Dict]: A list of dictionaries with URL spans.
    """
    spans = []
    url_regex = rb"[$|\W](https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*[-a-zA-Z0-9@%_\+~#//=])?)"
    text_bytes = post_text.encode("UTF-8")
    for match in re.finditer(url_regex, text_bytes):
        spans.append({
            "start": match.start(1),
            "end": match.end(1),
            "url": match.group(1).decode("UTF-8"),
        })
    return spans


def _parse_rich_urls(post_content: str):
    spans = []
    while True:
        span, post_content = _handle_first_rich_url(post_content)
        if span:
            spans.append(span)
        else:
            break
    return spans, post_content


def _handle_first_rich_url(post_content: str):
    regex = rb"\[(.*?)\]\(\s*(https?://[^\s)]+)\s*\)"
    text_bytes = post_content.encode("UTF-8")
    match = re.search(regex, text_bytes)
    if match:
        span = {
            "start": match.start(1) - 1,
            "end": match.end(1) - 1,
            "url": match.group(2).decode("UTF-8"),
        }
        post_content = (
                post_content[: match.start(1) - 1]
                + post_content[match.start(1): match.end(1)]
                + post_content[match.end():]
        )
        return span, post_content
    return None, post_content


def parse_facets(post_content: str):
    facets = []
    # Parse rich text URLs
    spans, post_content = _parse_rich_urls(post_content)
    for rich_url in spans:
        facets.append({
            "index": {
                "byteStart": rich_url["start"],
                "byteEnd": rich_url["end"],
            },
            "features": [
                {
                    "$type": "app.bsky.richtext.facet#link",
                    "uri": rich_url["url"],
                }
            ],
        })
    return facets, post_content


async def post_from_buffer():
    """
    Post a random message from the buffer to Bluesky.

    Returns:
        dict: The status of the posting operation.
    """
    if not BUFFER:
        logging.info("No posts in the buffer to post.")
        return {"status": "Buffer is empty"}

    post_content = random.choice(BUFFER)
    try:
        jwt_token = await bluesky_login()  # Ensure this function is defined elsewhere

        post_data = {
            "collection": "app.bsky.feed.post",
            "repo": BLUESKY_HANDLE,
            "record": {
                "type": "app.bsky.feed.post",
                "text": post_content,
                "createdAt": datetime.utcnow().isoformat() + "Z",
                "langs": ["th", "en-US"],
            },
        }

        # Parse and update facets in the post data
        facets, new_content = parse_facets(post_content)
        post_data["record"]["facets"] = facets
        post_data["record"]["text"] = new_content

        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {jwt_token}"}
            response = await client.post(
                "https://bsky.social/xrpc/com.atproto.repo.createRecord",
                json=post_data,
                headers=headers,
            )
            response.raise_for_status()

        BUFFER.remove(post_content)
        logging.info(f"Successfully posted: {post_content}")
        return {"status": "Posted successfully", "post": post_content}

    except Exception as e:
        logging.error(f"Failed to post to Bluesky: {e}")
        return {"status": "Failed to post", "error": str(e)}

# Scheduler setup
scheduler = AsyncIOScheduler()
scheduler.add_job(post_from_buffer, CronTrigger(hour=7, minute=0))
scheduler.add_job(get_posts, CronTrigger(hour=0, minute=0))
scheduler.start()


def get_config(file: str) -> configparser.ConfigParser:
    """Read the configuration file."""
    config = configparser.ConfigParser()
    config.read(file)
    return config


@click.command()
@click.option("--config-file", "-a", type=click.Path(), default="config.ini")
@click.option("--config-profile", "-c", default="TEST", help="Select a config profile")
def main(config_file, config_profile):
    """Main function to start the application."""
    global BLUESKY_HANDLE, BLUESKY_PASSWORD

    config = get_config(config_file)
    BLUESKY_HANDLE = config[config_profile]["BLUESKY_HANDLE"]
    BLUESKY_PASSWORD = config[config_profile]["BLUESKY_PASSWORD"]
    port = config[config_profile].getint("PORT", 8000)

    logging.getLogger("uvicorn.access").addFilter(lambda record: "GET /health" not in record.getMessage())

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "alive"}


if __name__ == "__main__":
    main()
