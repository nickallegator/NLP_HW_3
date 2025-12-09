import random, json
import nltk
from nltk.corpus import brown
from sklearn.preprocessing import LabelEncoder

NUM_SENTENCES = 500
random.seed(42)

def extract_contexts():
    nltk.download("brown")
    nltk.download("universal_tagset")

    tagged = brown.tagged_sents(categories="news", tagset="universal")

    selected = []
    while len(selected) < NUM_SENTENCES:
        s = random.choice(tagged)
        tags = [t for _, t in s]
        noun_pos = [i for i, t in enumerate(tags) if t == "NOUN"]
        if any(i >= 3 for i in noun_pos):
            selected.append(s)

    contexts = []
    for s in selected:
        tags = [t for _, t in s]
        for i in range(3, len(tags)):
            if tags[i] == "NOUN":
                contexts.append(tags[i-3:i])

    flat_tags = [t for ctx in contexts for t in ctx]
    le = LabelEncoder()
    le.fit(flat_tags)

    encoded = [le.transform(ctx).tolist() for ctx in contexts]
    with open("q6_contexts.json", "w") as f:
        json.dump(encoded, f, indent=2)

    print("Saved", len(encoded), "encoded chunks to q6_contexts.json")

extract_contexts()
