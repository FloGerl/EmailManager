
# Email Manager

Welcome to my first public project!

This Python script, emailmanager.py, is designed to automate email management using OpenAI's GPT API. The script utilizes the Gmail API to fetch emails, creates prompts based on the email content, and sends these prompts to the GPT API. Based on the GPT's response, the script then performs an appropriate action such as archiving, deleting, replying to, or forwarding the email.

## Features

1. **Fetch Emails:** The script fetches emails using Gmail's API and organizes them into a manageable format.
2. **Generate Prompts:** For each email, a prompt is generated based on its content. This prompt is then sent to the GPT API.
3. **Interact with GPT:** The GPT model processes the prompt and returns an action to be taken.
4. **Perform Action:** Based on the GPT's response, the script performs the appropriate action on the email.

For the moment, the archive and delete functions only applies a label to the email.

## Prerequisites

Before running the script, make sure you have the following:

- Python 3.11 or later
- Access to Gmail's API (you will need to generate your own credentials file)
- An API key from OpenAI

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/FloGerl/EmailManager.git
    ```
2. Navigate to the project directory:
    ```
    cd your-repository
    ```
3. Install the required Python dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage

You can run the script from the command line as follows:

```
python emailmanager.py
```

You can also use the `--all` flag to process all emails:

```
python emailmanager.py --all
```

To skip confirmation prompts, use the `--skip-confirmation` flag:

```
python emailmanager.py --skip-confirmation
```

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## License

This project is licensed under the terms of the MIT license.
