# Bluesky PRIDE Bot

A FastAPI-based bot for the Bluesky social platform that automatically posts summarized dataset information from PRIDE Archive. The bot schedules posts throughout the day and manages a buffer of datasets to ensure regular updates.

## Features

- Uses the [FLAN-T5 small model](https://huggingface.co/google/flan-t5-small) for summarizing dataset descriptions.
- Posts to Bluesky five times a day at scheduled times.
- Manages a daily buffer to avoid repetitive posting.
- Configurable with environment-specific settings.

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install dependencies:**
    ```bash
    conda env create -f environment.yml
    ```

3. **Configure environment variables:**
    Ensure the `BLUESKY_HANDLE` and `BLUESKY_PASSWORD` variables are set in the `config.ini` file or as environment variables.

4. **Create a config.ini file:**
    Example `config.ini` file:
    ```ini
    [DEFAULT]
    PORT = 8000

    [TEST]
    BLUESKY_HANDLE = "your_bluesky_handle"
    BLUESKY_PASSWORD = "your_bluesky_password"
    PORT = 8000
    ```

## Usage

### Run the Bot

Start the bot using the following command:

```bash
python main.py --config-file config.ini --config-profile TEST
