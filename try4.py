# pip install spacy
# python -m spacy download en_core_web_sm
import re
import spacy

nlp = spacy.load("en_core_web_sm")

NEG_TOKENS = {"not", "n't", "never", "no"}

def already_negated(sent):
    return any(t.dep_ == "neg" or t.lower_ in NEG_TOKENS for t in sent)

def subject_token(root):
    for c in root.children:
        if c.dep_ in ("nsubj", "nsubjpass", "expl"):
            return c
    return None

def choose_do_form(root, subj):
    # Tense/person heuristic based on POS tags
    tag = root.tag_
    if tag == "VBD":
        return "did"
    if tag == "VBZ":
        return "does"
    # Imperative / present plural -> "do"
    # If subject is 3rd-person singular noun/pronoun and verb tag is present (VBP/VB), we could pick "does",
    # but VBZ already covered; default to "do" here.
    return "do"

def insert_after(tokens, spaces, idx, word):
    tokens.insert(idx + 1, word)
    spaces.insert(idx + 1, " ")

def apply_contractions(text):
    # Basic, safe contractions (optional). Extend as needed.
    rules = [
        (r"\b(C|c)an not\b", r"\1an't"),
        (r"\b(W|w)ill not\b", r"\1on't"),
        (r"\b(D|d)o not\b", r"\1on't"),
        (r"\b(D|d)oes not\b", r"\1oesn't"),
        (r"\b(D|d)id not\b", r"\1idn't"),
        (r"\b(I|i)s not\b", r"\1sn't"),
        (r"\b(A|a)re not\b", r"\1ren't"),
        (r"\b(W|w)as not\b", r"\1asn't"),
        (r"\b(W|w)ere not\b", r"\1eren't"),
        (r"\b(H|h)ave not\b", r"\1aven't"),
        (r"\b(H|h)as not\b", r"\1asn't"),
        (r"\b(H|h)ad not\b", r"\1adn't"),
        (r"\b(S|s)hould not\b", r"\1houldn't"),
        (r"\b(W|w)ould not\b", r"\1ouldn't"),
        (r"\b(C|c)ould not\b", r"\1ouldn't"),
        (r"\b(M|m)ust not\b", r"\1ustn't"),
        (r"\b(M|m)ight not\b", r"\1ightn't"),
        (r"\b(S|s)hall not\b", r"\1han't"),
    ]
    for pat, rep in rules:
        text = re.sub(pat, rep, text)
    return text

def negate_one_sentence(sent, use_contractions=False, preserve_existing=True):
    # If already negated, optionally leave as-is
    if preserve_existing and already_negated(sent):
        return sent.text_with_ws

    root = sent.root
    toks = [t.text for t in sent]
    spcs = [t.whitespace_ for t in sent]

    # Find auxiliaries/copula attached to the root (ordered by position)
    auxes = [c for c in root.children if c.dep_ in ("aux", "auxpass", "cop") or c.tag_ == "MD"]
    auxes.sort(key=lambda t: t.i)

    # Case 1: we have an aux/modal/copula -> insert "not" after the first
    if auxes:
        anchor = auxes[0]
        idx = anchor.i - sent[0].i
        insert_after(toks, spcs, idx, "not")
        out = "".join(t + s for t, s in zip(toks, spcs))
        return apply_contractions(out) if use_contractions else out

    # Case 2: Imperative ("Open the door.")
    if root.tag_ == "VB" and subject_token(root) is None:
        # Insert "Do not" before the root; keep root in base form
        idx = root.i - sent[0].i
        # Capitalize "Do" if sentence starts with capital
        cap = sent[0].text[:1].isupper()
        helper = "Do" if cap else "do"
        toks.insert(idx, helper)
        spcs.insert(idx, " ")
        insert_after(toks, spcs, idx, "not")
        # ensure the main verb is base form (it already is in imperatives)
        out = "".join(t + s for t, s in zip(toks, spcs))
        return apply_contractions(out) if use_contractions else out

    # Case 3: Do-support
    subj = subject_token(root)
    helper = choose_do_form(root, subj)
    idx = root.i - sent[0].i
    # Replace main verb with its lemma (base)
    base = root.lemma_ if root.lemma_ else root.text
    toks[idx] = base
    # Insert helper and "not" before the verb
    toks.insert(idx, helper)
    spcs.insert(idx, " ")
    insert_after(toks, spcs, idx, "not")
    out = "".join(t + s for t, s in zip(toks, spcs))
    return apply_contractions(out) if use_contractions else out

def negate(text, use_contractions=False, preserve_existing=True):
    """
    Negate each sentence in `text`.
    - use_contractions: turn "is not" -> "isn't", "do not" -> "don't", etc.
    - preserve_existing: if a sentence is already negated, leave it unchanged.
    """
    doc = nlp(text)
    pieces = []
    for sent in doc.sents:
        pieces.append(negate_one_sentence(sent, use_contractions, preserve_existing))
    return "".join(pieces)

# ----------------- Demo -----------------
if __name__ == "__main__":
    examples = [
        "model life cycle procedure",
        # "They went to the office.",
        # "She is happy.",
        # "We have finished the report.",
        # "He can swim fast.",
        # "Open the door.",
        # "There is a book on the table.",
        # "I will submit the validation memo tomorrow.",
        # "The model underperforms on long sentences.",
        # "I am ready.",
        # "She has to leave now.",
    ]
    for e in examples:
        print(f"{e} -> {negate(e, use_contractions=True)}")
