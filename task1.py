"""
SupportAI - Task 1
FAQ Knowledge Base Foundation

This module contains:
1. FAQ knowledge base
2. Keyword-based FAQ search
3. FAQ lookup by ID
4. FAQ lookup by category
5. Demonstration tests
"""


# ============================================================
# 1. FAQ KNOWLEDGE BASE
# ============================================================

faqs = [
    {
        "id": "faq-001",
        "category": "Account",
        "question": "How do I reset my password?",
        "answer": (
            "Click 'Forgot Password' on the login page. "
            "Enter your registered email address and check your inbox "
            "for a reset link valid for 24 hours."
        ),
        "keywords": ["password", "reset", "forgot", "login"]
    },

    {
        "id": "faq-002",
        "category": "Billing",
        "question": "What is your refund policy?",
        "answer": (
            "We offer full refunds within 30 days of purchase for "
            "unused subscriptions. Partial refunds are available for "
            "annual plans cancelled after 30 days."
        ),
        "keywords": ["refund", "money back", "cancel", "billing"]
    },

    {
        "id": "faq-003",
        "category": "Shipping",
        "question": "How long does shipping take?",
        "answer": (
            "Standard shipping takes 5–7 business days. Express shipping "
            "(2–3 business days) is available at checkout for an "
            "additional fee."
        ),
        "keywords": ["shipping", "delivery", "tracking", "express"]
    },

    {
        "id": "faq-004",
        "category": "Account",
        "question": "How do I update my email address?",
        "answer": (
            "Go to Settings → Account → Email. Enter your new email "
            "and confirm via the verification link sent to the new address."
        ),
        "keywords": ["email", "update", "change", "account"]
    },

    {
        "id": "faq-005",
        "category": "Technical",
        "question": "The app keeps crashing. What should I do?",
        "answer": (
            "First, update to the latest version from the App Store or "
            "Google Play. If the issue persists, clear the app cache in "
            "Settings → Storage, then restart your device."
        ),
        "keywords": ["crash", "bug", "error", "technical", "app"]
    }
]


# ============================================================
# 2. SEARCH FAQS BY KEYWORD
# ============================================================

def search_by_keyword(faqs, query):
    """
    Search FAQs using words from the user's query.

    The search is case-insensitive. Query words are compared
    against the FAQ keywords, question, and category.

    FAQs are ranked by the number of matching query words.
    The FAQ with the highest number of matches appears first.

    Args:
        faqs (list): List of FAQ dictionaries.
        query (str): User's search question.

    Returns:
        list: Matching FAQs ordered by match score.
    """

    # Convert the user's query to lowercase and split it into words.
    query_words = query.lower().split()

    matching_faqs = []

    # Check every FAQ in the knowledge base.
    for faq in faqs:

        # Convert FAQ fields to lowercase so the search
        # works regardless of uppercase/lowercase letters.
        question = faq["question"].lower()
        category = faq["category"].lower()

        # Convert all keywords to lowercase.
        keywords = [keyword.lower() for keyword in faq["keywords"]]

        score = 0

        # Check every word from the user's query.
        for word in query_words:

            # Ignore very small/common words.
            # They usually do not help identify an FAQ.
            if len(word) < 3:
                continue

            # Check whether the query word matches:
            # 1. a keyword
            # 2. the question
            # 3. the category
            keyword_match = any(word in keyword for keyword in keywords)
            question_match = word in question
            category_match = word in category

            if keyword_match or question_match or category_match:
                score += 1

        # Only keep FAQs where at least one query word matched.
        if score > 0:
            matching_faqs.append((score, faq))

    # Sort from highest score to lowest score.
    matching_faqs.sort(key=lambda item: item[0], reverse=True)

    # Return only the FAQ dictionaries, not the scores.
    return [faq for score, faq in matching_faqs]


# ============================================================
# 3. GET FAQ BY ID
# ============================================================

def get_faq_by_id(faqs, faq_id):
    """
    Find an FAQ using its unique ID.

    Args:
        faqs (list): List of FAQ dictionaries.
        faq_id (str): ID of the FAQ to find.

    Returns:
        dict | None: Matching FAQ dictionary, or None if not found.
    """

    for faq in faqs:

        if faq["id"].lower() == faq_id.lower():
            return faq

    return None


# ============================================================
# 4. GET FAQS BY CATEGORY
# ============================================================

def get_faqs_by_category(faqs, category):
    """
    Return all FAQs belonging to a given category.

    The category comparison is case-insensitive.

    Args:
        faqs (list): List of FAQ dictionaries.
        category (str): Category to search for.

    Returns:
        list: FAQs belonging to the requested category.
    """

    matching_faqs = []

    for faq in faqs:

        if faq["category"].lower() == category.lower():
            matching_faqs.append(faq)

    return matching_faqs


# ============================================================
# 5. DISPLAY SEARCH RESULTS
# ============================================================

def display_search_results(query):
    """
    Search the FAQ knowledge base and display the results
    in a clear format.
    """

    print(f"\nQuery: {query}")

    results = search_by_keyword(faqs, query)

    if not results:
        print("  No matching FAQs found.")
        return

    for faq in results:
        print(f"  [{faq['category']}] {faq['question']}")
        print(f"  → {faq['answer']}")
        print()


# ============================================================
# 6. DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SupportAI - Task 1")
    print("FAQ Knowledge Base Foundation")
    print("=" * 60)

    # Test 1: Password search
    display_search_results("forgot my password")

    # Test 2: Refund search
    display_search_results("refund")

    # Test 3: Query that should have no match
    display_search_results("weather today")

    # --------------------------------------------------------
    # Test get_faq_by_id()
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Testing get_faq_by_id()")
    print("=" * 60)

    faq = get_faq_by_id(faqs, "faq-001")

    if faq:
        print(f"ID: {faq['id']}")
        print(f"Category: {faq['category']}")
        print(f"Question: {faq['question']}")
        print(f"Answer: {faq['answer']}")
    else:
        print("FAQ not found.")

    # --------------------------------------------------------
    # Test get_faqs_by_category()
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("Testing get_faqs_by_category()")
    print("=" * 60)

    account_faqs = get_faqs_by_category(faqs, "account")

    for faq in account_faqs:
        print(f"[{faq['category']}] {faq['question']}")