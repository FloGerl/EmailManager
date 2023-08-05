"""A simple chatbot class that uses OpenAI's GPT API to generate responses."""

import datetime
import logging
import sys
import tkinter as tk
from tkinter import filedialog

import openai
import tiktoken
import yaml


class OpenAIAPI:
    """Handles authentication and configuration for the OpenAI API.

    Get a license key from https://platform.openai.com/account/api-keys
    """

    def __init__(self) -> None:
        """Reads the OpenAI credentials from a YAML file."""

        with open("credentials.yaml", "r", encoding="UTF-8") as credentials_file:
            config = yaml.safe_load(credentials_file)

        if "openai" in config and "api_key" in config["openai"]:
            openai.api_key = config["openai"]["api_key"]
        else:
            logging.error(
                "Please enter your OpenAI API key in the credentials.yaml file as follows:\n"
                "openai:\n"
                "  api_key : sk-YOURKEYHERE"
            )
            raise ValueError("OpenAI API key not found in credentials file.")


class ChatBot:
    """A simple chatbot class that uses OpenAI's GPT API to generate responses."""

    TOKEN_LIMITS = {
        "gpt-3.5-turbo": 4097,  # 4097 total tokens per request
        "gpt-3.5-turbo-16k": 16384,  # 16385 total tokens per request
        "gpt-4": 8192,  # 8193 total tokens per request
        # "gpt-4-32k": 32768,  # 32769 total tokens per request
    }
    TOKEN_PRICES = {
        "gpt-3.5-turbo": 0.002 * 0.001,  # $0.002 / 1K tokens
        "gpt-3.5-turbo-16k": 0.004 * 0.001,  # $0.004 / 1K tokens
        "gpt-4": 0.06 * 0.001,  # $0.06 / 1K tokens
        # "gpt-4-32k": 0.12 * 0.001,  # $0.12 / 1K tokens
    }

    MODELS = set(("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"))

    def __init__(
        self, system: str = "", model: str = "gpt-3.5-turbo", available_function: list[dict[str, str]] | None = None
    ) -> None:
        """Initializes the chatbot."""
        OpenAIAPI()  # Initialises the OpenAI API with the api-key
        self.system: str = system
        self.model: str = model
        self.messages: list[dict[str, str]] = []
        self.functions: list[dict[str, str]] = [] if available_function is None else available_function
        self.__consumed_tokens: int = 0
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def update_available_functions(self, available_function: list[dict[str, str]]) -> None:
        """Update the available functions."""
        self.functions = available_function

    def __call__(self, message: str) -> str | dict[str, str]:
        """Add a message and execute the chatbot and return the AI's answer."""
        if message.strip():
            self.messages.append({"role": "user", "content": message})
            execution_result: str | dict[str, str] = self.execute()
            if isinstance(execution_result, dict):
                result: str = f"Function call: {execution_result['function_call']['name']}({execution_result['function_call']['arguments']})"
            if isinstance(execution_result, str):
                result: str = self.execute()
            self.messages.append({"role": "assistant", "content": result})
            return execution_result

        return "Please enter a message."

    def execute(self) -> str | dict[str, str]:
        """Execute the chatbot and return the response."""
        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                functions=self.functions,
                function_call="auto",
            )
            self.__consumed_tokens += completion.usage["total_tokens"]
            message = completion.choices[0].message
            if message.get("function_call"):
                return message
                # return f"Function call: {message['function_call']['name']}({message['function_call']['arguments']})"
                # return f"Function call: {message['function_call']})"
            return str(message.content)
        except openai.error.AuthenticationError:
            logging.error("Invalid API key. Please check your configuration.")
            return "I'm sorry, but I am experiencing technical difficulties right now. Please try again later."
        except openai.error.APIError as error:
            logging.error(f"OpenAI API error: {str(error)}")
            return "I'm sorry, but I am experiencing technical difficulties right now. Please try again later."
        except openai.error.OpenAIError:
            logging.error("API call failed. Please check your internet connection.")
            return "I'm sorry, but I am experiencing technical difficulties right now. Please try again later."

    def get_messages(self) -> list[dict[str, str]]:
        """Return the messages."""
        return self.messages

    def save_messages(self, filename: str = "") -> None:
        """Save the messages to a file."""

        if not filename:
            filename = f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        try:
            with open(filename, "w", encoding="UTF-8") as file:
                for message in self.messages:
                    file.write(f"{message['role']}: {message['content']}\n")
        except IOError as io_error:
            logging.error(f"Failed to save chat history: {io_error}")

    def get_total_tokens_used(self) -> int:
        """Return the total number of tokens used."""
        return self.__consumed_tokens

    @staticmethod
    def get_tokens_for_message(messages: str | dict[str, str]) -> int:
        """Return the number of tokens required for a message."""
        if isinstance(messages, str):
            messages = {"content": messages}

        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(message)) for _, message in messages.items())


def chat(model="gpt-3.5-turbo") -> None:
    """A simple chatbot that uses OpenAI's GPT-3 API to generate responses."""
    chatbox = ChatBot(system="I am a chatbot. Ask me a question.", model=model)

    while True:
        print("Please answer:")
        my_input = input()

        if my_input in ("exit", "quit", "q"):
            # Save all the messages to a file
            chatbox.save_messages()
            sys.exit()

        if my_input == "load file":
            try:
                # Open a file selector dialog to choose the file to load
                root = tk.Tk()
                root.withdraw()
                file_path = filedialog.askopenfilename()

                print("Please enter a request to prepend to the file content:")
                file_header = input()

                total_input = file_header + "\n"
                with open(file_path, "r", encoding="UTF-8") as file:
                    total_input += file.read()

                print(f"This is my input: {total_input}")
                print(chatbox(total_input))
            except ImportError:
                logging.warning("GUI libraries are not available. Please enter the file path directly.")
            except FileNotFoundError:
                logging.error("File not found. Please try again.")
        else:
            print(chatbox(my_input))


if __name__ == "__main__":
    try:
        OpenAIAPI()

        model_list: list[openai.api_resources.model.Model] = openai.Model.list()["data"]
        model_name_list = [model["id"] for model in model_list]

        chat(model="gpt-4")

    except Exception as e:
        logging.error(f"Failed to initialize chatbot: {e}")
