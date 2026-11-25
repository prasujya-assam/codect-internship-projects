"""
Movie Recommendation System
===========================
Implements two classic approaches on a built-in sample ratings dataset:

  1. Collaborative Filtering (item-based, cosine similarity)
     "Users who liked the movies you liked also liked..."
  2. Content-Based Filtering (genre similarity)
     "Other movies with genres similar to your favorites..."

Pure Python, no external libraries. Run:  python movie_recommender.py
"""

import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# 1. Sample dataset: 10 movies and ratings from 8 users (1-5 scale)
# ---------------------------------------------------------------------------
MOVIES = {
    "The Matrix":          {"genres": ["sci-fi", "action"]},
    "Inception":           {"genres": ["sci-fi", "thriller"]},
    "Interstellar":        {"genres": ["sci-fi", "drama"]},
    "The Dark Knight":     {"genres": ["action", "crime"]},
    "Titanic":             {"genres": ["romance", "drama"]},
    "The Notebook":        {"genres": ["romance", "drama"]},
    "Superbad":            {"genres": ["comedy"]},
    "The Hangover":        {"genres": ["comedy"]},
    "Toy Story":           {"genres": ["animation", "family", "comedy"]},
    "Finding Nemo":        {"genres": ["animation", "family"]},
}

# user -> {movie: rating}
RATINGS = {
    "Alice":   {"The Matrix": 5, "Inception": 5, "Interstellar": 4, "The Dark Knight": 4},
    "Bob":     {"The Matrix": 4, "The Dark Knight": 5, "Titanic": 2, "Superbad": 5, "The Hangover": 5},
    "Carol":   {"Titanic": 5, "The Notebook": 5, "Interstellar": 3, "Toy Story": 4},
    "Dave":    {"Superbad": 4, "The Hangover": 5, "The Matrix": 3, "Inception": 4},
    "Eve":     {"Toy Story": 5, "Finding Nemo": 5, "The Notebook": 3, "Titanic": 3},
    "Frank":   {"Inception": 5, "The Matrix": 5, "The Dark Knight": 5, "The Hangover": 2},
    "Grace":   {"The Notebook": 5, "Titanic": 4, "Finding Nemo": 4, "Toy Story": 5},
    "Heidi":   {"Interstellar": 5, "Inception": 4, "The Notebook": 1, "Superbad": 2},
}


# ---------------------------------------------------------------------------
# 2. Collaborative filtering: item-item cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    """Cosine similarity between two rating vectors (dicts of movie->rating)."""
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[m] * b[m] for m in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def build_item_similarity_matrix(ratings):
    """Compute cosine similarity between every pair of items,
    based on which users rated both of them."""
    items = sorted({m for user_ratings in ratings.values() for m in user_ratings})
    sims = defaultdict(dict)
    for i, item_a in enumerate(items):
        for item_b in items[i + 1:]:
            vec_a = {u: r[item_a] for u, r in ratings.items() if item_a in r}
            vec_b = {u: r[item_b] for u, r in ratings.items() if item_b in r}
            sim = cosine_similarity(vec_a, vec_b)
            sims[item_a][item_b] = sim
            sims[item_b][item_a] = sim
    return sims


def recommend_collaborative(user, ratings, sims, n=5):
    """Score unseen items by weighted average of the user's own ratings,
    weighted by item-item similarity."""
    user_ratings = ratings[user]
    scores = defaultdict(float)
    sim_sums = defaultdict(float)
    for seen_movie, rating in user_ratings.items():
        for candidate, sim in sims[seen_movie].items():
            if candidate in user_ratings:
                continue
            scores[candidate] += sim * rating
            sim_sums[candidate] += abs(sim)
    predicted = {
        m: scores[m] / sim_sums[m] for m in scores if sim_sums[m] > 0
    }
    return sorted(predicted.items(), key=lambda x: -x[1])[:n]


# ---------------------------------------------------------------------------
# 3. Content-based filtering: genre Jaccard similarity
# ---------------------------------------------------------------------------
def genre_similarity(movie_a, movie_b):
    ga = set(MOVIES[movie_a]["genres"])
    gb = set(MOVIES[movie_b]["genres"])
    union = ga | gb
    return len(ga & gb) / len(union) if union else 0.0


def recommend_content_based(user, ratings, n=5):
    """Score unseen items by average genre similarity to the movies
    the user rated highly (4 stars and up)."""
    user_ratings = ratings[user]
    liked = [m for m, r in user_ratings.items() if r >= 4]
    scores = []
    for candidate, meta in MOVIES.items():
        if candidate in user_ratings:
            continue
        if liked:
            avg_sim = sum(genre_similarity(candidate, m) for m in liked) / len(liked)
        else:
            avg_sim = 0.0
        scores.append((candidate, avg_sim))
    return sorted(scores, key=lambda x: -x[1])[:n]


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------
def print_recommendations(title, recs):
    print(f"\n{title}")
    print("-" * 55)
    if not recs:
        print("  (no candidates left — user has rated everything)")
        return
    for movie, score in recs:
        genres = ", ".join(MOVIES[movie]["genres"])
        print(f"  * {score:.2f}  {movie}  ({genres})")


def main():
    print("=" * 60)
    print("MOVIE RECOMMENDER — collaborative + content-based filtering")
    print("=" * 60)
    print(f"\nDataset: {len(MOVIES)} movies, {len(RATINGS)} users, "
          f"{sum(len(r) for r in RATINGS.values())} ratings")

    sims = build_item_similarity_matrix(RATINGS)

    print("\nSample item similarities (collaborative, cosine):")
    for pair in [("The Matrix", "Inception"), ("Titanic", "The Notebook"),
                 ("The Matrix", "The Hangover")]:
        a, b = pair
        print(f"  {a} <-> {b}: {sims[a][b]:.2f}")

    for user in ["Alice", "Bob"]:
        print(f"\n{'=' * 60}\nRecommendations for {user} "
              f"(already rated: {', '.join(sorted(RATINGS[user]))})")
        recs_cf = recommend_collaborative(user, RATINGS, sims, n=3)
        recs_cb = recommend_content_based(user, RATINGS, n=3)
        print_recommendations("[Collaborative filtering]", recs_cf)
        print_recommendations("[Content-based]", recs_cb)

    print("\n" + "=" * 60)
    print("Try your own profile! Enter ratings as  movie:score  pairs.")
    print("Example:  The Matrix:5, Titanic:2, Superbad:4   (empty line quits)")
    while True:
        try:
            entry = input("\nYour ratings> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not entry:
            break
        profile = {}
        ok = True
        for part in entry.split(","):
            if ":" not in part:
                print("  ! Invalid format, use  movie:score  separated by commas")
                ok = False
                break
            name, _, val = part.partition(":")
            name, val = name.strip(), val.strip()
            if name not in MOVIES or not val.replace(".", "", 1).isdigit():
                print(f"  ! Unknown movie '{name}' or bad score '{val}'. "
                      f"Available: {', '.join(MOVIES)}")
                ok = False
                break
            profile[name] = min(5.0, max(1.0, float(val)))
        if not ok or not profile:
            continue

        temp_ratings = dict(RATINGS)
        temp_ratings["You"] = profile
        temp_sims = build_item_similarity_matrix(temp_ratings)
        print_recommendations("[Collaborative filtering]",
                              recommend_collaborative("You", temp_ratings, temp_sims, n=5))
        print_recommendations("[Content-based]",
                              recommend_content_based("You", temp_ratings, n=5))


if __name__ == "__main__":
    main()