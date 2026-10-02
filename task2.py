"""
SupportAI - Task 2
OpenRouter LLM Integration

This module:
1. Reuses the FAQ knowledge base from Task 1.
2. Searches for the best matching FAQ.
3. Sends the FAQ and user's question to OpenRouter.
4. Generates a natural support response.
"""

import os

import requests
from dotenv import load_dotenv

from task1 import faqs, search_by_keyword


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = "openai/gpt-4o-mini"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# LLM CLIENT
# ============================================================

class LLMClient:
    """
    Client for communicating with the OpenRouter LLM API.
    """

    def __init__(self, api_key, model):
        """
        Initialize the LLM client.

        Args:
            api_key (str): OpenRouter API key.
            model (str): Model name to use.
        """

        self.api_key = api_key
        self.model = model

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    def generate(self, prompt, system_message=None, max_tokens=512):
        """
        Send a chat-completion request to OpenRouter.

        Args:
            prompt (str): User prompt sent to the LLM.
            system_message (str, optional): Instructions that
                control how the LLM should respond.
            max_tokens (int): Maximum number of output tokens.

        Returns:
            str: The LLM-generated response.

        Raises:
            ValueError: If the API key is missing.
            RuntimeError: If the API request fails.
        """

        # Make sure the API key exists before making the request.
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Please add it to your .env file."
            )

        # OpenRouter expects the API key in a Bearer token.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Create the messages required by the chat API.
        messages = []

        if system_message:
            messages.append(
                {
                    "role": "system",
                    "content": system_message
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # Request body sent to OpenRouter.
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=60
            )

        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not connect to OpenRouter: {error}"
            ) from error

        # Check whether OpenRouter returned an HTTP error.
        if not response.ok:
            try:
                error_details = response.json()
            except ValueError:
                error_details = response.text

            raise RuntimeError(
                f"OpenRouter API request failed "
                f"(HTTP {response.status_code}): {error_details}"
            )

        # Convert the successful response into Python data.
        try:
            response_data = response.json()

            return response_data["choices"][0]["message"]["content"]

        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"Unexpected response format from OpenRouter: {error}"
            ) from error

    # --------------------------------------------------------
    # GENERATE FAQ RESPONSE
    # --------------------------------------------------------

    def generate_faq_response(self, user_question, faq_entry):
        """
        Generate a natural support response using an FAQ entry.

        The LLM is instructed to stay strictly within the
        information provided by the official FAQ.

        Args:
            user_question (str): Question asked by the user.
            faq_entry (dict): FAQ selected by the search system.

        Returns:
            str: Natural-language support response.
        """

        # System prompt controls the behavior of the LLM.
        system_message = """
You are SupportAI, a friendly and professional customer-support assistant.

Your job is to answer the user's question using ONLY the information
contained in the provided official FAQ.

Rules:
1. Do not invent facts, policies, prices, dates, features, or procedures.
2. Do not add information that is not present in the FAQ.
3. Rewrite the official answer in a natural and helpful way.
4. Keep the response friendly and professional.
5. Keep the response under 150 words.
6. If the FAQ does not fully answer the user's question, clearly say
   that the available FAQ does not provide enough information.
7. Do not pretend to know information that is not in the FAQ.
"""

        # Build the user prompt containing the source FAQ
        # and the actual question.
        prompt = f"""
Official FAQ question:
{faq_entry["question"]}

Official FAQ answer:
{faq_entry["answer"]}

User's question:
{user_question}

Using only the official FAQ information above, write a concise,
friendly support response to the user.
"""

        return self.generate(
            prompt=prompt,
            system_message=system_message,
            max_tokens=512
        )


# ============================================================
# FIND BEST FAQ
# ============================================================

def find_best_faq(user_question):
    """
    Find the highest-ranked FAQ for a user's question.

    This function reuses search_by_keyword() from Task 1.

    Args:
        user_question (str): Question asked by the user.

    Returns:
        dict | None: Best matching FAQ, or None if there is no match.
    """

    results = search_by_keyword(faqs, user_question)

    if not results:
        return None

    return results[0]


# ============================================================
# DEMONSTRATION
# ============================================================

def demonstrate_question(client, user_question):
    """
    Demonstrate the complete Task 2 workflow for one question.

    Workflow:
    User question
        ↓
    Task 1 keyword search
        ↓
    Best FAQ
        ↓
    OpenRouter LLM
        ↓
    Natural support response
    """

    print("\n" + "=" * 70)
    print(f"Question: {user_question}")
    print("=" * 70)

    # Step 1: Find the best FAQ using Task 1.
    matched_faq = find_best_faq(user_question)

    if matched_faq is None:
        print("\nNo matching FAQ found.")
        return

    # Step 2: Display the matched FAQ.
    print(f"\nMatched FAQ: {matched_faq['question']}")

    # Step 3: Give the FAQ and user question to the LLM.
    try:
        response = client.generate_faq_response(
            user_question=user_question,
            faq_entry=matched_faq
        )

        # Step 4: Display the generated response.
        print("\nSupportAI Response:")
        print(response)

    except (ValueError, RuntimeError) as error:
        print(f"\nError: {error}")


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SupportAI - Task 2")
    print("OpenRouter LLM Integration")
    print("=" * 70)

    # Make sure the API key is available.
    if not OPENROUTER_API_KEY:
        print("\nERROR:")
        print("OPENROUTER_API_KEY is missing.")
        print("Create a .env file and add:")
        print("OPENROUTER_API_KEY=your_api_key_here")

    else:

        # Create the OpenRouter client.
        client = LLMClient(
            api_key=OPENROUTER_API_KEY,
            model=DEFAULT_MODEL
        )

        # Test Question 1
        demonstrate_question(
            client,
            "I can't remember my login password"
        )

        # Test Question 2
        demonstrate_question(
            client,
            "Can I get my money back for my subscription?"
        )

