"""A script to manage my emails"""

import argparse
import base64
import json
import logging
import os
import os.path
import pickle
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import getaddresses
from enum import Enum, auto

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from chatbot import ChatBot

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
]


class ResponseOptions(Enum):
    """The possible options regarding the GPT answer."""

    ACCEPTED = auto()
    RETRY = auto()
    IGNORE = auto()
    ERROR = auto()
    FIRST = auto()


@dataclass
class Message:
    """A class to represent a message"""

    message_id: str
    timestamp: datetime
    content: str
    is_unread: bool
    tokens: int = 0
    attachments: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post-initialisation function used to calculate the number of tokens this message is worth"""
        self.tokens = ChatBot.get_tokens_for_message(self.__str__())

    def get_dict(self, include_content: bool = False) -> dict[str, str | datetime | bool]:
        """Returns the message as a dict"""
        message_dict = {"message_id": self.message_id, "timestamp": self.timestamp, "is_unread": self.is_unread}
        if include_content:
            message_dict["content"] = self.content

        return message_dict

    def __str__(self) -> str:
        """Return a string representation of the message"""
        return (
            f"Message(message_id={self.message_id}, "
            f"timestamp={self.timestamp}, "
            f"is_unread={self.is_unread}, "
            f"content={self.content})"
            f"attachemnt_names={self.attachments}"
        )


@dataclass
class EmailThread:
    """Class to represent an email thread"""

    thread_id: str
    subject: str
    participants: set[str]

    label_ids: set[str]
    messages: list[Message] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a string representation of the email thread"""
        return (
            f"EmailThread(threadId={self.thread_id}, "
            f"subject={self.subject}, "
            f"participants={self.participants}, "
            f"labelIds={self.label_ids}, "
            f"messages={self.messages})"
        )

    def get_dict(self, include_message_content: bool = False):
        """Returns the content of the class as a dict"""
        thread_dict = {
            "thread_id": self.thread_id,
            "subject": self.subject,
            "participants": self.participants,
            "label_ids": self.label_ids,
            "messages": [],
        }
        for message in self.messages:
            thread_dict["messages"].append(message.get_dict(include_message_content))

        return thread_dict

    def get_json(self, include_message_content: bool = False) -> str:
        """Returns the content of the class as a JSON string"""
        return json.dumps(self.get_dict(include_message_content=include_message_content), default=str)

    def get_tokens(self, include_message_content: bool = False) -> int:
        """Get the number of tokens for the whole thread"""
        return ChatBot.get_tokens_for_message(self.get_json(include_message_content=include_message_content))


class GMailManager:
    """A Class to manage a GMail inbox automatically."""

    TOKENS_FACTOR = 1.1

    def __init__(self) -> None:
        """Initialise the GMailManager"""

        self.__service = self.__initialise_service()
        self.__own_email = self.get_user_email()
        self.__label_dict = {label["name"]: label["id"] for label in self.fetch_labels()}
        self.__email_cache: list[EmailThread] = []
        self.__last_update: datetime = datetime.now() - timedelta(days=10)

    def __initialise_service(self) -> Resource:
        """Initialise a session with the Gmail API.

        It retrieves the credentials from the credentials.json file and creates a token.pickle file.

        Returns: Resource usable for Gmail API calls.

        """
        creds = None
        if os.path.exists("token.pickle"):
            with open("token.pickle", "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("gmail_credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)

        service: Resource = build("gmail", "v1", credentials=creds)

        return service

    def get_user_email(self):
        """Get the email address of the user"""
        try:
            # Get the user's profile information
            profile = self.__service.users().getProfile(userId="me").execute()  # pylint: disable=E1101

            # The email address is in the 'emailAddress' field
            return profile["emailAddress"]
        except HttpError as error:
            print(f"An error occurred: {error}")

    def update_email_cache(self) -> None:
        """Update the email cache with the latest emails."""

        if len(self.__email_cache) == 0:
            self.__last_update = datetime.now() - timedelta(days=10)

        new_threads = self.fetch_threads_since_dt(self.__last_update)

        # Add new threads to the cache
        self.__email_cache.extend(new_threads)

        # Remove duplicates and keep the newest thread in case of duplication
        thread_dict = {thread.thread_id: thread for thread in self.__email_cache}
        self.__email_cache = list(thread_dict.values())

        # Sort by the newest message
        self.__email_cache.sort(key=lambda thread: thread.messages[0].timestamp, reverse=True)

        self.__last_update = datetime.now()

    def run_all(self, needs_confirmation: bool = True) -> bool:
        """Treat all emails."""

        while self.run(needs_confirmation):
            pass

        return True

    def run(self, needs_confirmation: bool = True) -> bool:
        """Treat the latest unread email."""

        # Step 1. Load email cache
        self.update_email_cache()

        # Step 2. Retrieve latest unread email
        # currently_managed_email = self.fetch_latest_unread_email()
        currently_managed_email = self.get_next_thread_to_treat()
        if currently_managed_email is None:
            logging.info("No new messages.")
            return False
        # logging.info(f"Treating '{currently_managed_email.subject}'")

        # Step 3. Retrieve the associated threads
        # Step 3.1 Retrieve the associated participants
        participants = self.get_participants_from_emailthread(currently_managed_email)

        # Step 3.2 Retrieve the associated threads
        emails_from_participants: list[EmailThread] = self.fetch_threads_from_participants(participants)
        # for participant in participants:
        # emails_from_participants.extend(self.fetch_emails_from_participant(participant))

        r_o = ResponseOptions.FIRST
        response: str = ""
        while r_o not in [ResponseOptions.ACCEPTED, ResponseOptions.IGNORE]:
            # Step 4. Create and send the prompt
            prompt = self.create_prompt(currently_managed_email, emails_from_participants, r_o, response)
            # save prompt
            with open("prompt.txt", "w", encoding="UTF-8") as prompt_file:
                prompt_file.write(prompt)
            response = self.send_prompt(prompt)
            if response is None:
                logging.error("No GPT response. Quitting")
                sys.exit()

            # Step 5. Act upon the response
            logging.info(f"Acting on email with subject '{currently_managed_email.subject}'")
            if isinstance(response, dict):
                # Function call
                r_o, human_feedback = self.does_function_call(
                    email_thread=currently_managed_email,
                    function_call=response["function_call"],
                    confirm_decision=needs_confirmation,
                )
                if needs_confirmation:  # If a human has intervened, let's take the opportunity to update the prompt
                    # self.update_prompt_instructions(
                    #     last_prompt=prompt, last_answer=response, human_decision=human_feedback
                    # )
                    pass
            else:
                logging.info(f"Response: {response}")
                break

        return True

    def fetch_labels(self) -> list[dict[str, str]]:
        """Get all labels"""
        tries = 0

        while tries < 3:
            try:
                results = self.__service.users().labels().list(userId="me").execute()  # pylint: disable=E1101
                break
            except TimeoutError:
                tries += 1
                logging.error("Timeout error.")

        labels = results.get("labels", [])
        # logging.info(f"Labels: {labels}")
        return labels

    def fetch_threads_since_dt(self, after_dt: datetime) -> list[EmailThread]:
        """Helper function to get all emails from the last n days."""
        timestamp = int(after_dt.timestamp())
        query = f"after:{timestamp}"
        results = self.__service.users().threads().list(userId="me", q=query).execute()  # pylint: disable=E1101
        threads = results.get("threads", [])
        return self.threads_to_email_threads(threads)

    def get_participants_from_emailthread(self, email_thread: EmailThread) -> list[str]:
        """Get the participants of an email thread"""
        participants = email_thread.participants
        if self.__own_email is not None:
            participants = [p for p in participants if p != self.__own_email]
        return participants

    def get_next_thread_to_treat(self) -> EmailThread | None:
        """Get the next email to treat from the cached emails"""

        # Define the GPTd label ID for checking
        gptd_label_id = self.get_label_id("GPTd")  # Update this with the actual ID of the GPTd label
        inbox_label_id = self.get_label_id("INBOX")

        for thread in self.__email_cache:
            # Skip threads that have the GPTd label
            if gptd_label_id in thread.label_ids:
                continue

            # Skip anything that no longer is in the inbox
            if inbox_label_id not in thread.label_ids:
                continue

            # Check if the thread has an unread message
            for message in thread.messages:
                # If the UNREAD label ID is in the message label IDs, this is the next email to treat
                if message.is_unread:
                    return thread

        # If no untreated email is found, return None
        return None

    def apply_label(self, message_id: str, label_name: str):
        """Apply a label to a message"""
        label_id = self.get_label_id(label_name)
        if label_id is None:
            logging.error(f"Label '{label_name}' not found.")
            return

        # Apply the label to the message
        self.__service.users().messages().modify(
            userId="me", id=message_id, body={"addLabelIds": [label_id]}
        ).execute()  # pylint: disable=E1101
        # logging.info(f"Applied label '{label_name}' to message {message_id}.")

    def threads_to_email_threads(self, threads: list[dict[str, str]]) -> list[EmailThread]:
        """Convert a list of thread data into a list of EmailThread instances"""

        email_threads = []

        for thread in threads:
            thread_full = (
                self.__service.users().threads().get(userId="me", id=thread["id"]).execute()
            )  # pylint: disable=E1101
            messages = thread_full.get("messages", [])

            # Initialize the lists and sets that will be used to create the EmailThread
            message_objs = []
            participants = set()
            label_ids = set()

            for message in messages:
                # Get the timestamp and content
                timestamp = datetime.fromtimestamp(int(message["internalDate"]) // 1000)

                content = ""
                attachment_names: list[str] = []
                payload = message["payload"]
                for part in payload.get("parts", []):
                    # Check if the part is an attachment
                    if part["filename"]:
                        # If the part has a filename, add it to the list of attachment names
                        attachment_names.append(part["filename"])
                    if part["mimeType"] in ["text/plain", "text/html", "multipart/alternative"]:
                        body = part["body"]
                        try:
                            data = part["parts"][0]["body"]["data"] if body["size"] == 0 else body["data"]
                        except KeyError:
                            data = ""
                            logging.error(f"No content to this message: {message['id']}")
                        text = base64.urlsafe_b64decode(data).decode("utf-8")
                        if part["mimeType"] == "text/html":
                            text = self.get_text_from_html(text)  # remove HTML entities
                        content = text

                # Check if the message is unread
                is_unread = "UNREAD" in message["labelIds"]

                # Add the message to the list of Message objects
                message_objs.append(
                    Message(
                        message_id=message["id"],
                        timestamp=timestamp,
                        content=content,
                        is_unread=is_unread,
                        attachments=attachment_names,
                    )
                )

                # Get the participants and labels
                for header in message["payload"]["headers"]:
                    name = header["name"]
                    value = header["value"]
                    if name.lower() == "subject":
                        subject = value
                    elif name.lower() in ["from", "to", "cc", "bcc"]:
                        addresses = getaddresses([value])  # parse the email addresses
                        for addr in addresses:
                            if addr[1] != self.__own_email:
                                participants.add(addr[1])

                # Get the labels
                label_ids.update(set(message["labelIds"]))

            # Sort the message objects by timestamp from newest to oldest
            message_objs.sort(key=lambda x: x.timestamp, reverse=True)

            # Create the EmailThread
            email_thread = EmailThread(
                thread_id=thread["id"],
                subject=subject,
                participants=participants,
                label_ids=label_ids,
                messages=message_objs,
            )

            # Add the EmailThread to the list
            email_threads.append(email_thread)

        return email_threads

    def fetch_threads_from_participants(
        self, participants: list[str], max_threads: int = 10, oldest_thread_months: int = 6
    ) -> list[EmailThread]:
        """Get the threads that include the given participants"""

        query = f"from:({') OR from:('.join(participants)}) newer_than:{oldest_thread_months}m"
        results = (
            self.__service.users().threads().list(userId="me", q=query, maxResults=max_threads).execute()
        )  # pylint: disable=E1101
        threads = results.get("threads", [])
        return self.threads_to_email_threads(threads)

    def create_prompt(
        self,
        latest_unread_email: EmailThread,
        participants_threads: list[EmailThread],
        response: ResponseOptions,
        previous_result: str = "",
    ) -> str:
        """Create a prompt for the GPT-3 model"""

        prompt = ""
        # Maximum tokens for the largest available model
        max_tokens = max(ChatBot.TOKEN_LIMITS.values())
        # max_tokens = ChatBot.TOKEN_LIMITS['gpt-4']
        # logging.info(f"Maximum tokens: {max_tokens}")

        # Copy the cached_emails list
        cached_threads = self.__email_cache.copy()

        # Reduce the number of emails to include to match the token limit
        # Low weights: important
        weights = {
            "Current Thread": 1,
            "Participants": 3,
            "Other Threads": 5,
        }
        while True:
            tokens = {
                "Current Thread": latest_unread_email.get_tokens(True),
                "Participants": sum([thread.get_tokens() for thread in participants_threads]),
                "Other Threads": sum([thread.get_tokens() for thread in cached_threads]),
            }
            weighted_tokens = {
                "Current Thread": weights["Current Thread"] * tokens["Current Thread"],
                "Participants": weights["Participants"] * tokens["Participants"],
                "Other Threads": weights["Other Threads"] * tokens["Other Threads"],
            }
            total_tokens = sum(tokens.values())
            if total_tokens * GMailManager.TOKENS_FACTOR * 1.2 < max_tokens:
                # logging.info(
                #     f"Accepting:\n"
                #     f"\t{latest_unread_email.get_tokens(True)} tokens for current thread\n"
                #     f"\t{sum([thread.get_tokens() for thread in participants_threads])} for participants threads\n"
                #     f"\t{sum([thread.get_tokens() for thread in cached_threads])} for other threads\n"
                #     f"\tTotal: {total_tokens} tokens"
                # )
                break
            # logging.info(
            #     f"Need to reduce token usage:\n"
            #     f"\t{latest_unread_email.get_tokens()} tokens for current thread\n"
            #     f"\t{sum([thread.get_tokens() for thread in participants_threads])} for participants threads\n"
            #     f"\t{sum([thread.get_tokens() for thread in cached_threads])} for other threads\n"
            #     f"\tTotal: {total_tokens} tokens"
            # )

            # If the prompt is too long:
            if weighted_tokens["Other Threads"] > max(
                weighted_tokens["Current Thread"], weighted_tokens["Participants"]
            ):
                # Remove one thread from the cached threads
                cached_threads.pop()
                # weighted_tokens["Other Threads"] = weights["Other Threads"] * sum(
                #     [thread.get_tokens() for thread in cached_threads]
                # )
            elif weighted_tokens["Participants"] > max(
                weighted_tokens["Current Thread"], weighted_tokens["Other Threads"]
            ):
                # Remove one thread from the participants threads
                participants_threads.pop()
                # weighted_tokens["Participants"] = weights["Participants"] * sum(
                #     [thread.get_tokens() for thread in participants_threads]
                # )
            else:
                # Remove one thread from the current thread
                latest_unread_email.messages.pop()
                # weighted_tokens["Current Thread"] = weights["Current Thread"] * latest_unread_email.get_tokens()

            # logging.info(f"Total tokens: {total_tokens}")

        latest_email_formatted = str(latest_unread_email)
        participants_threads_json = "\n".join([thread.get_json() for thread in participants_threads])
        historical_threads_json = "\n".join([thread.get_json() for thread in cached_threads])

        # Create the prompt
        prompt = "You are managing my e-mails. "
        if self.__own_email is not None:
            prompt += f"My email address is {self.__own_email}. "
        prompt += "Here is my latest email. What should I do with it?"
        prompt += f"--- Latest Email ---\n{latest_email_formatted}\n--- End of Latest Email ---\n"
        prompt += f"Here are some recent threads from the participants:\n{participants_threads_json}\n\n"
        prompt += (
            f"Here are some recent threads other than those with the same participants:\n{historical_threads_json}\n\n"
        )
        prompt += "Here are some instructions:\n"
        with open("prompt_instructions.txt", "r") as file:
            prompt += file.read()

        if response == ResponseOptions.RETRY:
            prompt += (
                "\nIMPORTANT: This is the second time you have been given this task. "
                "Your first answer was rejected and you should retry. "
                f"This was your first answer: {previous_result}.\n"
                "I would try another function call, or change the text if you tried replying or forwarding."
            )
        elif response == ResponseOptions.ERROR:
            prompt += (
                "\nIMPORTANT: This is the second time you have been given this task. "
                "Your first answer was rejected because the function call you gave does not exist. "
                f"This was your first answer: {previous_result}.\n"
                "I would try another function call, or change the text if you tried replying or forwarding."
            )

        with open("create_prompt.txt", "w", encoding="UTF-8") as prompt_file:
            prompt_file.write(prompt)

        return prompt

    def update_prompt_instructions(self, last_prompt, last_answer, human_decision):
        """Updates the promp_instructions.txt file based on the last user interaction."""

        with open("prompt_instructions.txt", "r", encoding="UTF-8") as prompt_instructions_file:
            prompt_instruction = prompt_instructions_file.read()

        # Step 1. Create the prompt for this feedback loop
        prompt = "You are managing my e-mails, and I am asking for your feedback. Here is the last prompt I sent you:"
        prompt += f"\n---\n{last_prompt}\n---\n"
        prompt += "Here is the last answer you gave me:"
        prompt += f"\n---\n{last_answer}\n---\n"
        prompt += f"The human in the loop decided that your answer was: {human_decision}.\n"
        prompt += (
            "You can now update the instructions I give you for next time, how would you change them, here they are:\n"
        )
        prompt += f"\n---\n{prompt_instruction}\n---\n"
        prompt += "Please use the function call to update the instructions, so that they will be updated for next time."
        prompt += "\nYou can also decide to keep the instructions as they are."

        available_functions = [
            {
                "name": "update_instructions",
                "description": "Updates the instructions for the email management prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instructions": {
                            "type": "string",
                            "description": "The instructions formatted as a string.",
                        },
                    },
                    "required": ["instructions"],
                },
            },
        ]

        # Step 2. Send the prompt to the API
        chatgpt = ChatBot(
            system="I improve instructions for a ChatGPT prompt.",
            model="gpt-3.5-turbo-16k",
            available_function=available_functions,
        )

        with open("instructions_corrections_prompt.txt", "w", encoding="UTF-8") as corrections_prompt_file:
            corrections_prompt_file.write(prompt)

        # Step 3. Get the response from the API
        response = chatgpt(prompt)
        logging.info(response)

        # Step 4. Update the prompt_instructions.txt file

    def get_label_id(self, label_name: str) -> str | None:
        """Fetch the ID of a label given its name"""
        return self.__label_dict.get(label_name, None)

    def send_prompt(self, prompt: str) -> str:
        """Send the prompt to the GPT-3 API"""
        prompt_tokens = ChatBot.get_tokens_for_message(prompt)

        model_order = ["gpt-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo"]
        model = None

        for model_name in model_order:
            if GMailManager.TOKENS_FACTOR * prompt_tokens < ChatBot.TOKEN_LIMITS[model_name]:
                model = model_name
                break

        if model is None:
            logging.error(f"The prompt is too long for all models: {prompt_tokens} tokens.")
            return

        # logging.info(f"Using model: {model} for {prompt_tokens} tokens.")

        chatbot = ChatBot(system="I am an email helper chatbot.", model=model, available_function=self.get_functions())
        response = chatbot(prompt)
        return response

    def get_functions(self) -> list[dict[str, str]]:
        """Build the functions dictionary for archive, delete, reply, forward."""

        available_functions = [
            {
                "name": "archive",
                "description": "Archive an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "The ID of the message to archive",
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "The reason for archiving the message - This can be a longer explanation, "
                                "you can also include the various elements in the prompt that made you "
                                "take this decision"
                            ),
                        },
                    },
                    "required": ["message_id", "reason"],
                },
            },
            {
                "name": "delete",
                "description": "Delete an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "The ID of the message to delete",
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "The reason for deleting the message - This can be a longer explanation, "
                                "you can also include the various elements in the prompt that made you "
                                "take this decision"
                            ),
                        }
                        # "permanently": {
                        #     "type": "boolean",
                        #     "description": "Whether to delete the message permanently",
                        # },
                    },
                    "required": ["message_id", "reason"],
                },
            },
            {
                "name": "reply",
                "description": "Reply to an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "The ID of the message to reply to",
                        },
                        "message_body": {
                            "type": "string",
                            "description": "The body of the reply message",
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "The reason for replying to the message - This can be a longer explanation, "
                                "you can also include the various elements in the prompt that made you "
                                "take this decision"
                            ),
                        },
                        "to_all": {
                            "type": "boolean",
                            "description": "Whether to reply to all participants",
                        },
                    },
                    "required": ["message_id", "message_body", "reason"],
                },
            },
            {
                "name": "forward",
                "description": "Forward an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "The ID of the message to forward",
                        },
                        "to_email": {
                            "type": "string",
                            "description": "The email address to forward to",
                        },
                        "message_body": {
                            "type": "string",
                            "description": "The body of the forward message",
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "The reason for forwarding the message - This can be a longer explanation, "
                                "you can also include the various elements in the prompt that made you "
                                "take this decision"
                            ),
                        },
                    },
                    "required": ["message_id", "to_email", "message_body", "reason"],
                },
            },
            {
                "name": "keep_as_is",
                "description": "Keeps the email as is, human intervention required.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "The ID of the message to forward",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason for keeping the message for the human.",
                        },
                    },
                    "required": ["message_id", "reason"],
                },
            },
            {
                "name": "unsubscribe",
                "description": "Unsubscribes from this newsletter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "string",
                            "description": "The ID of the message to unsubscribe from",
                        },
                        "reason": {
                            "type": "string",
                            "description": "The reason for unsubscribing",
                        },
                    },
                    "required": ["message_id", "reason"],
                },
            },
        ]

        return available_functions

    @staticmethod
    def get_text_from_html(html_content: str) -> str:
        """Extract text from HTML"""
        soup = BeautifulSoup(html_content, "html.parser")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()  # rip it out

        # get text
        text = soup.get_text()

        # replace carriage return with nothing
        text = re.sub("\r", "", text)

        # replace multiple newlines with a single newline
        text = re.sub("\n{2,}", "\n", text)

        # replace multiple tabs with a single tab
        text = re.sub("\t{2,}", "\t", text)

        return str(text).strip()

    def create_draft(self, message_body: str, thread_id: str = None):
        """Create a draft message"""
        message = {"raw": base64.urlsafe_b64encode(message_body.encode("utf-8")).decode("utf-8")}
        if thread_id:
            message["threadId"] = thread_id
        draft = {"message": message}
        try:
            result = self.__service.users().drafts().create(userId="me", body=draft).execute()  # pylint: disable=E1101
            logging.info(f"Draft id: {result['id']} created.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")

    def does_function_call(
        self, email_thread: EmailThread, function_call: str, confirm_decision: bool = True
    ) -> tuple[ResponseOptions, str]:
        """Apply the function call to the message"""
        thread_id = email_thread.thread_id
        message_id = email_thread.messages[0].message_id
        function_name = function_call["name"]
        arguments = function_call["arguments"]
        argument_dict = json.loads(arguments)

        logging.info(
            f"{'Suggesting' if confirm_decision else 'Applying'}:\t{function_name}\n\tBecause: {argument_dict['reason']}"
        )

        if confirm_decision:
            human_input = input("Is this correct? [y/yes/skip/retry/archive/delete/unsubscribe] ")
            if human_input.lower() == "retry":
                return ResponseOptions.RETRY, human_input
        else:
            human_input = "yes"

        new_labels: list[str] = []

        if human_input.lower() == "yes" or human_input.lower() == "y":
            if function_name == "archive":
                new_labels.append(self.label_archive(message_id))
            elif function_name == "delete":
                new_labels.append(self.label_delete(message_id))
            elif function_name == "reply":
                self.reply(message_id, argument_dict["message_body"], bool(argument_dict.get("to_all", False)))
            elif function_name == "forward":
                self.forward(message_id, argument_dict["to_email"], argument_dict["message_body"])
            elif function_name == "unsubscribe":
                new_labels.append(self.label_unsubscribe(message_id))
            elif function_name == "keep_as_is":
                pass
            else:
                logging.error(f"Unknown function {function_name}")
                return ResponseOptions.ERROR, human_input
        elif human_input.lower() == "retry":
            return ResponseOptions.RETRY, human_input
        elif human_input.lower() == "archive":
            new_labels.append(self.label_archive(message_id))
        elif human_input.lower() == "delete":
            new_labels.append(self.label_delete(message_id))
        elif human_input.lower() == "unsubscribe":
            new_labels.append(self.label_unsubscribe(message_id))

        # The email has been treated, we need to reflect this online and in the cache
        self.apply_label(message_id, "GPTd")  # Apply the label GPTd meaning that this email has been treated.
        new_labels.append("GPTd")

        # After treating the email, update the cache
        self.update_cache_after_treatment(thread_id, new_labels)

        if human_input.lower() == "skip":
            return ResponseOptions.IGNORE, human_input

        return ResponseOptions.ACCEPTED, human_input

    def update_cache_after_treatment(self, thread_id: str, new_labels: list[str] = None) -> None:
        """Update the cache after treating an email"""

        # Find the thread in the cache
        for thread in self.__email_cache:
            if thread.thread_id == thread_id:
                if new_labels is not None:
                    for new_label in new_labels:
                        new_label_id = self.get_label_id(new_label)
                        thread.label_ids.update(set([new_label_id]))
                return

    def reply(self, message_id: str, message_body: str, to_all: bool = False):
        """Compose a reply to an email"""
        # Get the original message
        msg_full = self.__service.users().messages().get(userId="me", id=message_id).execute()  # pylint: disable=E1101

        headers = msg_full["payload"]["headers"]

        if to_all:
            # Reply to all
            to_header = next(header for header in headers if header["name"].lower() == "to")
            cc_header = next((header for header in headers if header["name"].lower() == "cc"), {"value": ""})
            to_emails = f"{to_header['value']}, {cc_header['value']}"
        else:
            # Reply only to sender
            from_header = next(header for header in headers if header["name"].lower() == "from")
            to_emails = from_header["value"]

        # Create the message body
        message = MIMEMultipart()
        message["to"] = to_emails
        subject_header = next(header for header in headers if header["name"].lower() == "subject")
        message["subject"] = "Re: " + subject_header["value"]
        msg = MIMEText(message_body)
        message.attach(msg)

        self.create_draft(message.as_string(), msg_full["threadId"])

    def forward(self, message_id: str, to_email: str, message_body: str):
        """Compose a forward of an email"""
        # Get the original message
        original_message = (
            self.__service.users().messages().get(userId="me", id=message_id).execute()
        )  # pylint: disable=E1101

        subject = None
        for header in original_message["payload"]["headers"]:
            if header["name"].lower() == "subject":
                subject = header["value"]
                break

        # Check if subject is found
        if subject is None:
            logging.error("Could not find subject in original message.")
            return

        # Create the message body
        message = MIMEMultipart()
        message["to"] = to_email
        message["subject"] = "Fwd: " + subject
        msg = MIMEText(message_body)
        message.attach(msg)

        self.create_draft(message.as_string(), original_message["threadId"])

    def label_archive(self, message_id: str) -> str:
        """Add the label to archive the message, returns a list of applied labels."""
        self.apply_label(message_id, "ArchiveAfter1Day")
        return "ArchiveAfter1Day"

    def label_delete(self, message_id: str) -> str:
        """Add the label to delete the message, returns a list of applied labels."""
        self.apply_label(message_id, "DeleteAfter1Day")
        return "DeleteAfter1Day"

    def label_unsubscribe(self, message_id: str) -> str:
        """Add the label to delete the message, returns a list of applied labels."""
        self.apply_label(message_id, "NeedToUnsubscribe")
        return "NeedToUnsubscribe"

    def archive(self, message_id: str) -> None:
        """Archive an email"""
        self.__service.users().messages().modify(  # pylint: disable=E1101
            userId="me", id=message_id, body={"removeLabelIds": ["INBOX"]}
        ).execute()
        logging.info(f"Message archived. Id: {message_id}")

    def delete(self, message_id: str, permanently: bool = False) -> None:
        """Delete an email"""
        # Move the message to the trash
        self.__service.users().messages().trash(userId="me", id=message_id).execute()  # pylint: disable=E1101
        logging.info(f"Message with id: {message_id} has been trashed.")

        if permanently:
            # Delete it permanently
            self.__service.users().messages().delete(userId="me", id=message_id).execute()  # pylint: disable=E1101
            logging.info(f"Message with id: {message_id} has been deleted.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gmail_manager = GMailManager()

    # Define the argument parser
    parser = argparse.ArgumentParser(description="Manage GMail.")

    # Define the 'all' argument
    parser.add_argument("--all", dest="run_all", action="store_true", help="run all tasks")

    # Define the 'skip_confirmation' argument
    parser.add_argument(
        "--skip-confirmation", dest="needs_confirmation", action="store_false", help="skip confirmation prompts"
    )

    # Set defaults
    parser.set_defaults(run_all=False, needs_confirmation=True)

    # Parse the arguments
    args = parser.parse_args()

    if args.run_all:
        gmail_manager.run_all(args.needs_confirmation)
    else:
        gmail_manager.run(args.needs_confirmation)
