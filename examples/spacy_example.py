import spacy
from spacy import displacy

# Load the English model
nlp = spacy.load("en_core_web_md")

text = "Hello world! My name is John Smith and I live in New York City. I have 25 apples."

doc = nlp(text)

print("=== BASIC DOCUMENT INFO ===")
print(f"Text: {doc.text}")
print(f"Number of tokens: {len(doc)}")
print(f"Number of sentences: {len(list(doc.sents))}")

print("\n=== TOKENIZATION ===")
for token in doc:
    print(f"Token: '{token.text}' | Lemma: '{token.lemma_}' | POS: {token.pos_} | Tag: {token.tag_}")

print("\n=== NAMED ENTITY RECOGNITION ===")
for ent in doc.ents:
    print(f"Entity: '{ent.text}' | Label: {ent.label_} | Description: {spacy.explain(ent.label_)}")

print("\n=== DEPENDENCY PARSING ===")
for token in doc:
    print(f"'{token.text}' <-- {token.dep_} -- '{token.head.text}'")

print("\n=== SENTENCE SEGMENTATION ===")
for i, sent in enumerate(doc.sents, 1):
    print(f"Sentence {i}: {sent.text}")

print("\n=== TESTING SIMILARITY (requires vectors) ===")
# Test similarity between different texts
text1 = nlp("I love programming with Python")
text2 = nlp("Python programming is amazing")
text3 = nlp("The weather is nice today")

print(f"Similarity between 'I love programming with Python' and 'Python programming is amazing': {text1.similarity(text2):.3f}")
print(f"Similarity between 'I love programming with Python' and 'The weather is nice today': {text1.similarity(text3):.3f}")

print("\n=== TESTING LINGUISTIC FEATURES ===")
# Test various linguistic features
test_text = "The quick brown foxes are running quickly through the forest."
test_doc = nlp(test_text)

for token in test_doc:
    print(f"'{token.text}' | Lemma: {token.lemma_} | Is Alpha: {token.is_alpha} | Is Stop: {token.is_stop} | Is Punct: {token.is_punct}")

print("\n=== TESTING MATCHER PATTERNS ===")
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
# Define a pattern for finding proper nouns followed by common nouns
pattern = [{"POS": "PROPN"}, {"POS": "NOUN"}]
matcher.add("PROPER_NOUN_PHRASE", [pattern])

matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Found pattern: '{span.text}'")

print("\n=== TESTING CUSTOM PROCESSING ===")
# Process a longer text with more complex entities
complex_text = """
Apple Inc. is planning to release a new iPhone model in September 2024. 
The CEO Tim Cook announced this during a conference in San Francisco, California. 
The company's stock price increased by 5% after the announcement, reaching $180 per share.
Microsoft and Google are also expected to announce new products this year.
"""

complex_doc = nlp(complex_text)

print("Complex text entities:")
for ent in complex_doc.ents:
    print(f"'{ent.text}' ({ent.label_}) - {spacy.explain(ent.label_)}")

print("\n=== TOKEN ATTRIBUTES ANALYSIS ===")
sample_tokens = complex_doc[:10]  # First 10 tokens
for token in sample_tokens:
    print(f"Token: '{token.text}'")
    print(f"  - Lemma: {token.lemma_}")
    print(f"  - POS: {token.pos_} ({spacy.explain(token.pos_)})")
    print(f"  - Tag: {token.tag_} ({spacy.explain(token.tag_)})")
    print(f"  - Dependency: {token.dep_} ({spacy.explain(token.dep_)})")
    print(f"  - Is alphabetic: {token.is_alpha}")
    print(f"  - Is digit: {token.is_digit}")
    print(f"  - Is currency: {token.is_currency}")
    print("---")
