"""
SupportAI - Task 4
Conversational Helpdesk Agent

Integrates:
    Task 1 -> FAQ knowledge base
    Task 2 -> OpenRouter LLM
    Task 3 -> Hybrid FAQ matching

Features:
    - Multi-turn conversation
    - Confidence tracking
    - Human escalation
    - Mock ticket ID
    - Conversation history
    - Reset
    - Low-confidence streak
    - Interactive chat
    - End-to-end demonstration
"""

from dataclasses import dataclass
import random

from task1 import faqs
from task2 import LLMClient, OPENROUTER_API_KEY, DEFAULT_MODEL
from task3 import get_best_hybrid_match


# ============================================================
# CONVERSATION TURN
# ============================================================

@dataclass
class ConversationTurn:
    role: str
    content: str
    faq_id: str = None
    confidence: float = None


# ============================================================
# SUPPORT AGENT
# ============================================================

class SupportAgent:

    def __init__(
        self,
        faqs,
        llm_client,
        confidence_threshold=0.15
    ):
        """
        Initialize the SupportAI agent.

        Args:
            faqs: FAQ knowledge base from Task 1.
            llm_client: LLM client from Task 2.
            confidence_threshold: Minimum score required
                                  for accepting an FAQ match.
        """

        self.faqs = faqs
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

        # Conversation history
        self.conversation_history = []

        # Escalation state
        self.escalated = False

        # Consecutive low-confidence responses
        self.low_confidence_streak = 0

        # Latest match information
        self.last_faq_id = None
        self.last_confidence = None

    # ========================================================
    # HANDLE MESSAGE
    # ========================================================

    def handle_message(self, user_message):
        """
        Handle one user message.

        Flow:
            User message
                ↓
            Record message
                ↓
            Task 3 hybrid search
                ↓
            Check confidence
                ↓
            Task 2 LLM response
                ↓
            Record assistant response
        """

        # ----------------------------------------------------
        # 1. Record user message
        # ----------------------------------------------------

        user_turn = ConversationTurn(
            role="user",
            content=user_message
        )

        self.conversation_history.append(user_turn)

        # ----------------------------------------------------
        # 2. Run Task 3 hybrid search
        # ----------------------------------------------------

        result = get_best_hybrid_match(
            self.faqs,
            user_message,
            threshold=self.confidence_threshold
        )

        # ----------------------------------------------------
        # 3. No confident match
        # ----------------------------------------------------

        if result is None:

            self.low_confidence_streak += 1

            self.last_faq_id = None
            self.last_confidence = 0.0

            response = (
                "I don't have information about that in my "
                "knowledge base.\n"
                "Would you like me to connect you with a "
                "human support agent?"
            )

            # Suggest escalation after 3 consecutive
            # low-confidence responses.
            if self.low_confidence_streak >= 3:

                response += (
                    "\n\nI've been unable to find a confident "
                    "answer to your last three questions. "
                    "Type 'escalate' if you'd like to connect "
                    "with a human support agent."
                )

            assistant_turn = ConversationTurn(
                role="assistant",
                content=response,
                faq_id=None,
                confidence=0.0
            )

            self.conversation_history.append(
                assistant_turn
            )

            return response

        # ----------------------------------------------------
        # 4. Confident match found
        # ----------------------------------------------------

        faq, confidence = result

        # Reset low-confidence streak.
        self.low_confidence_streak = 0

        # Store latest FAQ information.
        self.last_faq_id = faq["id"]
        self.last_confidence = confidence

        # ----------------------------------------------------
        # 5. Generate response using Task 2
        # ----------------------------------------------------

        response = self.llm_client.generate_faq_response(
            user_question=user_message,
            faq_entry=faq
        )

        # ----------------------------------------------------
        # 6. Record assistant response
        # ----------------------------------------------------

        assistant_turn = ConversationTurn(
            role="assistant",
            content=response,
            faq_id=faq["id"],
            confidence=confidence
        )

        self.conversation_history.append(
            assistant_turn
        )

        return response

    # ========================================================
    # ESCALATE
    # ========================================================

    def escalate(
        self,
        reason="User requested human support"
    ):
        """
        Escalate the conversation to human support.
        """

        self.escalated = True

        # Generate mock ticket ID.
        ticket_id = (
            f"TICKET-{random.randint(10000, 99999)}"
        )

        response = (
            "Your request has been escalated to our "
            "support team.\n"
            f"Ticket ID: {ticket_id}\n"
            "Estimated response time: within 4 business hours."
        )

        # Store escalation in history.
        self.conversation_history.append(
            ConversationTurn(
                role="assistant",
                content=response,
                faq_id=None,
                confidence=None
            )
        )

        return response

    # ========================================================
    # CONVERSATION SUMMARY
    # ========================================================

    def get_conversation_summary(self):
        """
        Return a formatted summary of the conversation.
        """

        if not self.conversation_history:
            return "No conversation history."

        lines = []

        lines.append(
            "\n========== Conversation History =========="
        )

        for index, turn in enumerate(
            self.conversation_history,
            start=1
        ):

            role = turn.role.capitalize()

            lines.append(
                f"{index}. {role}: {turn.content}"
            )

            # Show FAQ ID and confidence for assistant answers.
            if turn.role == "assistant":

                if turn.faq_id is not None:

                    lines.append(
                        f"   FAQ ID: {turn.faq_id}"
                    )

                if turn.confidence is not None:

                    lines.append(
                        f"   Confidence: "
                        f"{turn.confidence:.2f}"
                    )

        lines.append(
            f"Escalated: "
            f"{'Yes' if self.escalated else 'No'}"
        )

        lines.append(
            "=========================================="
        )

        return "\n".join(lines)

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):
        """
        Clear conversation history and reset session state.
        """

        self.conversation_history.clear()

        self.escalated = False

        self.low_confidence_streak = 0

        self.last_faq_id = None

        self.last_confidence = None


# ============================================================
# WELCOME BANNER
# ============================================================

def print_banner():

    print()
    print(
        "╔══════════════════════════════════════════╗"
    )
    print(
        "║       SupportAI — Helpdesk Agent         ║"
    )
    print(
        "╚══════════════════════════════════════════╝"
    )
    print()

    print("Commands:")
    print("  history  - Display conversation history")
    print("  escalate - Connect to human support")
    print("  reset    - Clear conversation")
    print("  quit     - Exit the application")

    print()


# ============================================================
# DISPLAY RESPONSE
# ============================================================

def display_response(agent, response):

    if agent.last_faq_id is not None:

        print(
            "SupportAI "
            f"[confidence: "
            f"{agent.last_confidence:.2f}, "
            f"faq: {agent.last_faq_id}]:"
        )

    else:

        print(
            "SupportAI "
            f"[confidence: "
            f"{agent.last_confidence:.2f}]:"
        )

    print(response)
    print()


# ============================================================
# INTERACTIVE CHAT
# ============================================================

def run_chat(agent):

    print_banner()

    while True:

        try:

            user_message = input(
                "You: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\nThank you for using SupportAI. "
                "Goodbye!"
            )

            break

        except EOFError:

            print(
                "\nThank you for using SupportAI. "
                "Goodbye!"
            )

            break

        # Ignore empty input.
        if not user_message:
            continue

        command = user_message.lower()

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if command == "quit":

            print()
            print(
                "Thank you for using SupportAI. Goodbye!"
            )

            break

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        elif command == "history":

            print(
                agent.get_conversation_summary()
            )

            print()

        # ----------------------------------------------------
        # ESCALATE
        # ----------------------------------------------------

        elif command == "escalate":

            response = agent.escalate()

            print()
            print("SupportAI:")
            print(response)
            print()

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        elif command == "reset":

            agent.reset()

            print()
            print("SupportAI:")
            print(
                "Conversation reset. "
                "How can I help you?"
            )
            print()

        # ----------------------------------------------------
        # NORMAL USER QUESTION
        # ----------------------------------------------------

        else:

            try:

                response = agent.handle_message(
                    user_message
                )

                print()

                display_response(
                    agent,
                    response
                )

            except (ValueError, RuntimeError) as error:

                print()
                print("SupportAI:")
                print(
                    f"Sorry, I couldn't generate a response: "
                    f"{error}"
                )
                print()


# ============================================================
# END-TO-END DEMONSTRATION
# ============================================================

def run_demo(agent):

    print()
    print("=" * 70)
    print("SupportAI - End-to-End Demonstration")
    print("=" * 70)

    # --------------------------------------------------------
    # SCENARIO 1
    # --------------------------------------------------------

    print()
    print("SCENARIO 1: Clear FAQ question")
    print("-" * 70)

    question = "How do I reset my password?"

    print(f"You: {question}")

    response = agent.handle_message(
        question
    )

    display_response(
        agent,
        response
    )

    # --------------------------------------------------------
    # SCENARIO 2
    # --------------------------------------------------------

    print()
    print("SCENARIO 2: Paraphrased FAQ question")
    print("-" * 70)

    question = "I can't remember my login password"

    print(f"You: {question}")

    response = agent.handle_message(
        question
    )

    display_response(
        agent,
        response
    )

    # --------------------------------------------------------
    # SCENARIO 3
    # --------------------------------------------------------

    print()
    print("SCENARIO 3: Outside FAQ scope")
    print("-" * 70)

    question = "What are your office hours in Tokyo?"

    print(f"You: {question}")

    response = agent.handle_message(
        question
    )

    display_response(
        agent,
        response
    )

    # --------------------------------------------------------
    # SCENARIO 4
    # --------------------------------------------------------

    print()
    print("SCENARIO 4: User requests escalation")
    print("-" * 70)

    print("You: escalate")

    response = agent.escalate()

    print()
    print("SupportAI:")
    print(response)

    print()
    print("=" * 70)
    print("End-to-End Demonstration Complete")
    print("=" * 70)
    print()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print(
        "Initializing SupportAI..."
    )

    # --------------------------------------------------------
    # Check OpenRouter API key
    # --------------------------------------------------------

    if not OPENROUTER_API_KEY:

        print()
        print("ERROR:")
        print(
            "OPENROUTER_API_KEY is missing."
        )
        print(
            "Please make sure your .env file contains:"
        )
        print(
            "OPENROUTER_API_KEY=your_api_key_here"
        )

        return

    # --------------------------------------------------------
    # Create Task 2 LLM client
    # --------------------------------------------------------

    llm_client = LLMClient(
        api_key=OPENROUTER_API_KEY,
        model=DEFAULT_MODEL
    )

    # --------------------------------------------------------
    # Create SupportAgent
    # --------------------------------------------------------

    agent = SupportAgent(
        faqs=faqs,
        llm_client=llm_client,
        confidence_threshold=0.15
    )

    print(
        "SupportAI initialized successfully."
    )

    print()
    print(
        "Type 'demo' to run the four required scenarios."
    )
    print(
        "Type 'chat' to start the interactive helpdesk."
    )
    print()

    choice = input(
        "Choose demo/chat: "
    ).strip().lower()

    # --------------------------------------------------------
    # DEMO
    # --------------------------------------------------------

    if choice == "demo":

        run_demo(agent)

        print(
            "Starting interactive chat..."
        )

        run_chat(agent)

    # --------------------------------------------------------
    # CHAT
    # --------------------------------------------------------

    elif choice == "chat":

        run_chat(agent)

    # --------------------------------------------------------
    # INVALID INPUT
    # --------------------------------------------------------

    else:

        print()
        print(
            "Invalid choice."
        )
        print(
            "Starting interactive chat..."
        )

        run_chat(agent)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()