"""
SupportAI - Task 3
Intelligent FAQ Matching

This module:
1. Reuses the FAQ knowledge base from Task 1.
2. Builds a TF-IDF index.
3. Uses cosine similarity to match user questions.
4. Implements best-match and explanation functions.
5. Implements hybrid keyword + TF-IDF search.
6. Demonstrates the difference between all search methods.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from task1 import faqs, search_by_keyword


# ============================================================
# FAQ MATCHER
# ============================================================

class FAQMatcher:
    """
    Match user questions against FAQs using TF-IDF
    and cosine similarity.
    """

    def __init__(self, faqs):
        """
        Build a TF-IDF index from the FAQ questions and keywords.

        Args:
            faqs (list): List of FAQ dictionaries.
        """

        self.faqs = faqs

        # Combine each FAQ question and its keywords
        # into one searchable text document.
        self.corpus = [
            f"{faq['question']} {' '.join(faq['keywords'])}"
            for faq in faqs
        ]

        # Create the TF-IDF vectorizer.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )

        # Convert the FAQ corpus into TF-IDF vectors.
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.corpus
        )

    # --------------------------------------------------------
    # MATCH
    # --------------------------------------------------------

    def match(self, query, top_k=3):
        """
        Find the top FAQ matches for a user query.

        Args:
            query (str): User's question.
            top_k (int): Maximum number of results to return.

        Returns:
            list[tuple]: List of (faq_dict, confidence_score)
            tuples sorted by score descending.
        """

        # Convert user's query into a TF-IDF vector.
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarity between the query
        # and every FAQ.
        similarity_scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        )[0]

        # Create pairs containing FAQ and similarity score.
        scored_faqs = []

        for faq, score in zip(self.faqs, similarity_scores):

            # Convert NumPy value into normal Python float.
            confidence_score = round(float(score), 4)

            scored_faqs.append(
                (faq, confidence_score)
            )

        # Sort from highest similarity to lowest similarity.
        scored_faqs.sort(
            key=lambda item: item[1],
            reverse=True
        )

        # Return only requested number of results.
        return scored_faqs[:top_k]

    # --------------------------------------------------------
    # BEST MATCH
    # --------------------------------------------------------

    def best_match(self, query, threshold=0.15):
        """
        Return the best FAQ match if its similarity score
        meets or exceeds the threshold.

        Args:
            query (str): User's question.
            threshold (float): Minimum confidence required.

        Returns:
            tuple | None:
                (faq_dict, confidence_score) or None.
        """

        results = self.match(
            query,
            top_k=1
        )

        # No FAQs available.
        if not results:
            return None

        best_faq, best_score = results[0]

        # Accept result if it meets threshold.
        if best_score >= threshold:
            return best_faq, best_score

        return None

    # --------------------------------------------------------
    # EXPLAIN MATCH
    # --------------------------------------------------------

    def explain_match(self, query):
        """
        Return a formatted explanation of the top three
        FAQ matches and their similarity scores.

        Args:
            query (str): User's question.

        Returns:
            str: Formatted matching results.
        """

        results = self.match(
            query,
            top_k=3
        )

        lines = []

        for position, (faq, score) in enumerate(
            results,
            start=1
        ):
            lines.append(
                f"{position}. [{score:.4f}] "
                f"{faq['question']}"
            )

        return "\n".join(lines)


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(faqs, query, top_k=3):
    """
    Combine Task 1 keyword search with Task 3 TF-IDF matching.

    Keyword matches receive a base score of 0.5.
    TF-IDF results use their cosine similarity score.

    When the same FAQ appears in both methods, the higher
    score is kept.

    Args:
        faqs (list): List of FAQ dictionaries.
        query (str): User's question.
        top_k (int): Maximum number of results.

    Returns:
        list[tuple]:
            List of (faq_dict, score) tuples sorted by
            score in descending order.
    """

    # Dictionary used to store the highest score
    # for each FAQ.
    combined_results = {}

    # --------------------------------------------------------
    # STEP 1: KEYWORD SEARCH
    # --------------------------------------------------------

    keyword_results = search_by_keyword(
        faqs,
        query
    )

    for faq in keyword_results:

        # Keyword hits receive a base score of 0.5.
        combined_results[faq["id"]] = (
            faq,
            0.5
        )

    # --------------------------------------------------------
    # STEP 2: TF-IDF SEARCH
    # --------------------------------------------------------

    matcher = FAQMatcher(faqs)

    tfidf_results = matcher.match(
        query,
        top_k=top_k
    )

    for faq, score in tfidf_results:

        faq_id = faq["id"]

        # If FAQ already came from keyword search,
        # keep whichever score is higher.
        if faq_id in combined_results:

            existing_faq, existing_score = combined_results[
                faq_id
            ]

            if score > existing_score:
                combined_results[faq_id] = (
                    faq,
                    score
                )

        else:
            combined_results[faq_id] = (
                faq,
                score
            )

    # --------------------------------------------------------
    # STEP 3: SORT RESULTS
    # --------------------------------------------------------

    sorted_results = sorted(
        combined_results.values(),
        key=lambda item: item[1],
        reverse=True
    )

    return sorted_results[:top_k]


# ============================================================
# GET BEST HYBRID MATCH
# ============================================================

def get_best_hybrid_match(faqs, query, threshold=0.15):
    """
    Return the best FAQ from hybrid search if the score
    meets the confidence threshold.

    Args:
        faqs (list): List of FAQ dictionaries.
        query (str): User's question.
        threshold (float): Minimum confidence required.

    Returns:
        tuple | None:
            (faq_dict, confidence_score) or None.
    """

    results = hybrid_search(
        faqs,
        query,
        top_k=1
    )

    if not results:
        return None

    best_faq, best_score = results[0]

    if best_score >= threshold:
        return best_faq, best_score

    return None


# ============================================================
# DISPLAY KEYWORD SEARCH
# ============================================================

def display_keyword_results(query):
    """
    Display results from the Task 1 keyword search.
    """

    print("\n[Keyword Search]")

    results = search_by_keyword(
        faqs,
        query
    )

    if not results:
        print("  No matching FAQs found.")
        return

    for faq in results:
        print(
            f"  [{faq['category']}] "
            f"{faq['question']}"
        )


# ============================================================
# DISPLAY TF-IDF RESULTS
# ============================================================

def display_tfidf_results(matcher, query):
    """
    Display the top three TF-IDF matches.
    """

    print("\n[TF-IDF Matching]")

    results = matcher.match(
        query,
        top_k=3
    )

    for position, (faq, score) in enumerate(
        results,
        start=1
    ):
        print(
            f"  {position}. "
            f"[{score:.4f}] "
            f"{faq['question']}"
        )


# ============================================================
# DISPLAY HYBRID RESULTS
# ============================================================

def display_hybrid_results(query):
    """
    Display the top three hybrid search results.
    """

    print("\n[Hybrid Search]")

    results = hybrid_search(
        faqs,
        query,
        top_k=3
    )

    if not results:
        print("  No matching FAQs found.")
        return

    for position, (faq, score) in enumerate(
        results,
        start=1
    ):
        print(
            f"  {position}. "
            f"[{score:.4f}] "
            f"{faq['question']}"
        )


# ============================================================
# COMPARISON DEMONSTRATION
# ============================================================

def demonstrate_query(matcher, query):
    """
    Compare keyword search, TF-IDF matching,
    and hybrid search for one user query.
    """

    print("\n" + "=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    # Task 1 keyword search.
    display_keyword_results(query)

    # Task 3 TF-IDF search.
    display_tfidf_results(
        matcher,
        query
    )

    # Hybrid search.
    display_hybrid_results(query)

    # Best TF-IDF match.
    best = matcher.best_match(
        query,
        threshold=0.15
    )

    if best:
        faq, score = best

        print(
            f"\nBest TF-IDF match: "
            f"{faq['question']} "
            f"(confidence: {score:.4f})"
        )

    else:
        print(
            "\nBest TF-IDF match: "
            "No FAQ meets the confidence threshold."
        )

    # Best hybrid match.
    hybrid_best = get_best_hybrid_match(
        faqs,
        query,
        threshold=0.15
    )

    if hybrid_best:
        faq, score = hybrid_best

        print(
            f"Best Hybrid match: "
            f"{faq['question']} "
            f"(confidence: {score:.4f})"
        )

    else:
        print(
            "Best Hybrid match: "
            "No FAQ meets the confidence threshold."
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SupportAI - Task 3")
    print("Intelligent FAQ Matching")
    print("=" * 70)

    # Create TF-IDF matcher once.
    matcher = FAQMatcher(faqs)

    # Required demonstration queries.
    test_queries = [
        "I forgot my login credentials",
        "Can I get my money back?",
        "package delivery time"
    ]

    # Run all three search methods for every query.
    for query in test_queries:
        demonstrate_query(
            matcher,
            query
        )