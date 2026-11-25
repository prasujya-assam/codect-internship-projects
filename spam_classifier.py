"""
Spam Email Classifier (Naive Bayes)
====================================
A self-contained spam detection project. Trains a Multinomial Naive Bayes
classifier from scratch (pure Python, no external libraries) on a small
labeled SMS/email dataset, evaluates it, and lets you classify new messages.

Run:  python spam_classifier.py
"""

import math
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# 1. Training data: (label, text)  — 1 = spam, 0 = ham (legitimate)
# ---------------------------------------------------------------------------
TRAINING_DATA = [
    (1, "URGENT! You have won a 1 week FREE membership in our prize jackpot! Text WIN to 80085 now!"),
    (1, "Congratulations! You've been selected for a $1000 gift card. Claim your prize here www.claim-prize.biz"),
    (1, "FREE entry to win a brand new iPhone! Click http://win-big-now.com and claim yours today!"),
    (1, "Your mobile number has won £2,000,000 in our international lottery. Send your bank details to receive."),
    (1, "Hot singles in your area waiting to meet you! Sign up free now!!!"),
    (1, "You have 1 new message. Call 09061234567 to claim your cash prize. Calls cost £1.50/min"),
    (1, "Cheap meds online! Viagra, Cialis, Xanax with no prescription. Best prices guaranteed, buy now!"),
    (1, "WINNER! Our computer has picked YOU. To claim your $5000 reward reply YES immediately."),
    (1, "Exclusive offer just for you! Limited time 90% discount, act now or miss out forever!!!"),
    (1, "Your account will be suspended. Verify your credit card details at http://secure-verify-login.com now"),
    (1, "Earn $$$ from home! Work 1 hour a day, make $500 daily. No experience needed. Sign up today!"),
    (1, "Get rich quick! Guaranteed investment returns, double your money in 30 days, risk free!"),
    (1, "Final notice! Your loan application approved, no credit check, instant cash transfer today."),
    (1, "Click this link to unlock your free casino bonus chips. 500 free spins waiting for you!"),
    (1, "Text STOP to end. You are subscribed to premium ringtone service. £5 per week charged to your bill."),
    (0, "Hey, are we still meeting for lunch tomorrow at noon? Let me know!"),
    (0, "Can you send me the report before the meeting? Thanks a lot."),
    (0, "Hi Mom, I'll be home around 6pm. Do you need anything from the store?"),
    (0, "The project deadline has been moved to Friday. Please update your tasks accordingly."),
    (0, "Thanks for your email. I'll review the contract and get back to you by end of week."),
    (0, "Don't forget we have dinner reservations at 7pm tonight at the Italian place."),
    (0, "The flight lands at 14:30. I'll text you when I get to baggage claim."),
    (0, "Could you review my pull request when you get a chance? No rush until Monday."),
    (0, "Happy birthday! Hope you have a wonderful day celebrating with family."),
    (0, "Your order #45231 has shipped and should arrive within 3-5 business days."),
    (0, "Meeting notes from yesterday are attached. Let me know if I missed anything."),
    (0, "I'm running 10 minutes late, sorry! Start without me if I'm not there by 9."),
    (0, "The doctor's appointment is confirmed for Tuesday at 10am. Please arrive early."),
    (0, "Great job on the presentation today. The client was really impressed."),
    (0, "Reminder: gym membership payment due at the end of the month."),
]

TEST_DATA = [
    (1, "You've WON a free cruise! Claim your prize now at www.freecruise4u.com!!!"),
    (1, "URGENT: your bank account locked, confirm identity at http://fake-bank-secure.xyz"),
    (1, "Unbeatable deal! Cheap prescription pills delivered overnight, no prescription needed!"),
    (0, "Are you coming to Sarah's party on Saturday? Let me know by Thursday."),
    (0, "The quarterly results look good. Well done everyone on hitting our targets."),
    (0, "I left my charger at your place, can you bring it tomorrow?"),
]


# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z$0-9]+")

def tokenize(text):
    """Lowercase and split into simple word tokens."""
    text = text.lower()
    text = text.replace("!", " ! ").replace("$", " $ ")
    return TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# 3. Multinomial Naive Bayes (implemented from scratch)
# ---------------------------------------------------------------------------
class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha          # Laplace smoothing
        self.class_log_prior = {}
        self.word_log_probs = {}    # class -> {word: log P(word|class)}
        self.vocabulary = set()
        self._default = {}          # default log-prob for unseen words

    def fit(self, data):
        """data: list of (label, text)."""
        docs_per_class = defaultdict(list)
        for label, text in data:
            docs_per_class[label].append(tokenize(text))

        n_docs = len(data)
        for label, docs in docs_per_class.items():
            self.class_log_prior[label] = math.log(len(docs) / n_docs)

        word_counts = {label: Counter() for label in docs_per_class}
        total_words = Counter()
        for label, docs in docs_per_class.items():
            for tokens in docs:
                word_counts[label].update(tokens)
                total_words[label] += len(tokens)
                self.vocabulary.update(tokens)

        vocab_size = len(self.vocabulary)
        for label in word_counts:
            denominator = total_words[label] + self.alpha * vocab_size
            self.word_log_probs[label] = {
                w: math.log((c + self.alpha) / denominator)
                for w, c in word_counts[label].items()
            }
            self._default[label] = math.log(self.alpha / denominator)

    def predict(self, text):
        """Return (predicted_label, probabilities_per_class)."""
        tokens = [t for t in tokenize(text) if t in self.vocabulary]
        scores = {}
        for label, prior in self.class_log_prior.items():
            score = prior
            for tok in tokens:
                score += self.word_log_probs[label].get(tok, self._default[label])
            scores[label] = score
        max_score = max(scores.values())
        exp_scores = {l: math.exp(s - max_score) for l, s in scores.items()}
        total = sum(exp_scores.values())
        probs = {l: e / total for l, e in exp_scores.items()}
        return max(scores, key=scores.get), probs


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(model, data):
    correct = 0
    tp = fp = fn = 0                      # spam is the positive class
    for label, text in data:
        pred, _ = model.predict(text)
        if pred == label:
            correct += 1
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
    accuracy = correct / len(data)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return accuracy, precision, recall, f1


# ---------------------------------------------------------------------------
# 5. Main program
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SPAM EMAIL CLASSIFIER — Multinomial Naive Bayes (from scratch)")
    print("=" * 60)

    model = NaiveBayesClassifier(alpha=1.0)
    model.fit(TRAINING_DATA)
    print(f"\nTrained on {len(TRAINING_DATA)} messages "
          f"(vocabulary size: {len(model.vocabulary)} words)")

    acc, prec, rec, f1 = evaluate(model, TEST_DATA)
    print("\n--- Evaluation on held-out test set ---")
    print(f"Accuracy : {acc:.1%}")
    print(f"Precision: {prec:.1%}  (of flagged spam, how much really was spam)")
    print(f"Recall   : {rec:.1%}  (of real spam, how much we caught)")
    print(f"F1 score : {f1:.1%}")

    samples = [
        "WINNER! You have been selected for a free vacation. Reply NOW!",
        "Hi John, the team meeting is moved to 3pm tomorrow.",
        "Claim your $500 Walmart gift card by clicking this link today!",
        "Can you pick up milk on your way home? Thanks!",
    ]
    print("\n--- Classify new messages ---")
    for msg in samples:
        pred, probs = model.predict(msg)
        label = "SPAM" if pred == 1 else "HAM (not spam)"
        print(f"\nMessage : \"{msg}\"")
        print(f"Result  : {label}   (P(spam)={probs[1]:.1%}, P(ham)={probs[0]:.1%})")

    print("\n--- Try your own ---")
    print("Type a message and press Enter to classify it (empty line quits).")
    while True:
        try:
            text = input("\nYour message> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        pred, probs = model.predict(text)
        print(f"-> {'SPAM' if pred == 1 else 'HAM'}  "
              f"(confidence: {max(probs.values()):.1%})")


if __name__ == "__main__":
    main()