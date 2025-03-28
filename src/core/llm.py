import logging

from dotenv import dotenv_values
from langchain_ollama.chat_models import ChatOllama
from openai import OpenAI

ENV_VARS = dotenv_values(".env.local")
OPENAI_API_KEY = ENV_VARS["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM configuration and prompts using either OpenAI or Ollama.

    Parameters:
        model_name (str): The name of the model to use.
        provider (str): Which provider to use; either "openai" or "ollama".
        openai_api_key (str, optional): API key for OpenAI. Required if provider is "openai".
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        provider: str = "ollama",
        openai_api_key: str = None,
    ):
        self.model_name = model_name
        self.provider = provider.lower()

        if self.provider == "openai":
            if not openai_api_key:
                raise ValueError(
                    "OpenAI API key must be provided when using OpenAI as the provider."
                )

            self.client = OpenAI(api_key=openai_api_key)

        elif self.provider == "ollama":
            self.client = ChatOllama(model=model_name)

        else:
            raise ValueError(
                "Unsupported provider. Please use either 'openai' or 'ollama'."
            )

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generates a response for the given prompt using the selected LLM provider.

        Args:
            prompt (str): The human message or prompt.
            system_message (str, optional): A system instruction to set context. If None,
                a default (empty) system message is used for Ollama.

        Returns:
            str: The generated response from the LLM.
        """
        if self.provider == "openai":
            # Build the messages list in the format OpenAI expects.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            completion = self.client.chat.completions.create(
                model=self.model_name, messages=messages
            )
            return completion.choices[0].message.content

        elif self.provider == "ollama":
            messages = [("system", system_prompt), ("human", user_prompt)]
            ai_msg = self.client.invoke(messages)
            return ai_msg.content

        else:
            raise ValueError("Invalid provider configuration.")


if __name__ == "__main__":
    # Using the OpenAI provider:
    try:
        openai_manager = LLMManager(
            model_name="gpt-4o",
            provider="openai",
            openai_api_key=OPENAI_API_KEY,
        )
        response = openai_manager.generate_response(
            system_prompt="Write a one-sentence bedtime story about a unicorn.",
            user_prompt="You are a creative storyteller.",
        )
        print("OpenAI response:", response)
    except Exception as e:
        print("Error using OpenAI:", e)

    # Using the Ollama provider:
    try:
        ollama_manager = LLMManager(model_name="llama3.2", provider="ollama")
        response = ollama_manager.generate_response(
            system_prompt="Write a one-sentence bedtime story about a unicorn.",
            user_prompt="You are a creative storyteller.",
        )
        print("Ollama response:", response)
    except Exception as e:
        print("Error using Ollama:", e)
