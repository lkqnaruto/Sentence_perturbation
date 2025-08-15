# pip install -U transformers torch accelerate bitsandbytes
# (optional but recommended for selection) pip install -U sentencepiece
# Verifier: pip install -U transformers torch
# You must have access to: meta-llama/Meta-Llama-3-8B-Instruct (HF gated model)
# If needed: huggingface-cli login

import re
from typing import List, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from transformers import AutoTokenizer, __version__ as trv
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)

from transformers import AutoTokenizer


from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnOutClose(StoppingCriteria):
    def __init__(self, tokenizer):
        self.ids = tokenizer("</OUT>", add_special_tokens=False, return_tensors="pt").input_ids[0]
    def __call__(self, input_ids, scores, **kwargs):
        k = len(self.ids)
        if input_ids.shape[1] >= k and torch.equal(input_ids[0, -k:], self.ids.to(input_ids.device)):
            return True
        return False
    



# tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=True)
# good_template = tok.chat_template

# # Save to a text file for easy reuse
# with open("llama3_chat_template.jinja", "w", encoding="utf-8") as f:
#     f.write(good_template)

# print("Transformers:", trv)
# print(tok.chat_template)





# ---- Add near top of your file (after imports) ----
import re

# Explicit negation tokens / contractions
_EXPL_NEG_RE = re.compile(
    r"\b(?:not|never|no|none|nobody|nothing|nowhere|cannot|can't|won't|doesn't|don't|didn't|"
    r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|shouldn't|wouldn't|couldn't|mustn't|"
    r"mightn't|shan't)\b",
    re.IGNORECASE,
)

# Implicit negation patterns (safe, high-precision)
_IMPL_NEG_RES = [
    re.compile(r"\bno longer\b", re.IGNORECASE),
    re.compile(r"\bfail(?:s|ed|ing)?\s+to\b", re.IGNORECASE),
    re.compile(r"\brefus(?:e|es|ed|ing)\s+to\b", re.IGNORECASE),
    re.compile(r"\bdeclin(?:e|es|ed|ing)\s+to\b", re.IGNORECASE),
    re.compile(r"\bwithout\b", re.IGNORECASE),
    re.compile(r"\black(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bprohibit(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bforbid(?:s|den)?\b", re.IGNORECASE),
    re.compile(r"\bban(?:s|ned|ning)?\b", re.IGNORECASE),
    re.compile(r"\bavoid(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bprevent(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\brarely\b", re.IGNORECASE),
    re.compile(r"\bseldom\b", re.IGNORECASE),
    re.compile(r"\bhardly\b", re.IGNORECASE),
    re.compile(r"\bscarcely\b", re.IGNORECASE),
]

def _contains_explicit_negation(s: str) -> bool:
    return _EXPL_NEG_RE.search(s) is not None

def _contains_implicit_negation(s: str) -> bool:
    return any(p.search(s) for p in _IMPL_NEG_RES)

def _count_neg_markers(s: str) -> int:
    """
    Count negation markers (explicit + implicit) while avoiding double-counting 'no longer'.
    """
    t = s.lower()
    # Treat 'no longer' as a single marker
    t, n_no_longer = re.subn(r"\bno longer\b", "<NEG_PHRASE>", t)
    explicit = len(_EXPL_NEG_RE.findall(t))
    implicit = sum(len(p.findall(t)) for p in _IMPL_NEG_RES if p.pattern != r"\bno longer\b")
    return explicit + implicit + n_no_longer

def _is_double_negation(s: str) -> bool:
    # Heuristic: >=2 neg markers likely indicates double-negation (or overly negative wording)
    return _count_neg_markers(s) >= 2

# Add a helper to extract the payload safely (and fall back gracefully):
_OUT_RE = re.compile(r"<OUT>(.*?)</OUT>", re.DOTALL)

def _extract_out(s: str) -> str:
    m = _OUT_RE.search(s)
    if m:
        return m.group(1).strip()
    # Fallbacks: first line/sentence with an explicit/implicit negation
    s = s.strip()
    first_line = s.splitlines()[0]
    # If the first line has something like "Note:", drop that prefix
    first_line = re.sub(r"^\s*(?:note|disclaimer)\s*:\s*", "", first_line, flags=re.I)
    # If we still don't have negation markers, try splitting on period and picking the first clause with negation
    if not (_contains_explicit_negation(first_line) or _contains_implicit_negation(first_line)):
        for piece in re.split(r"(?<=[.!?])\s+", s):
            if _contains_explicit_negation(piece) or _contains_implicit_negation(piece):
                return piece.strip()
    return first_line


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NEG_TOKENS = {
    "not", "n't", "never", "no", "cannot", "can't", "won't", "doesn't",
    "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't",
    "couldn't", "mustn't", "mightn't", "shan't"
}

SYSTEM_PROMPT = (
    "You are a precise editor. Negate the user's short sentence or query with the FEWEST possible edits. "
    "Treat keyword-like inputs (e.g., 'download SAR filings') as imperatives and output a grammatical negated form "
    "('Do not download SAR filings.'). Preserve tense/person and keep ALL entities (names, acronyms, dates, numbers, "
    "codes, file names, paths) unchanged. If <KEEP>...</KEEP> tags appear, copy their contents verbatim and remove "
    "the tags in your answer. Output ONLY the negated sentence, nothing else."
)

# Replace SYSTEM_PROMPT with:
SYSTEM_PROMPT = (
    "You are a function. Your job is to negate the user's short sentence or query "
    "with the FEWEST possible edits. Treat keyword-like inputs (e.g., 'download SAR filings') "
    "as imperatives and output a grammatical negated form ('Do not download SAR filings.'). "
    "Preserve tense/person and keep ALL entities (names, acronyms, dates, numbers, codes, file names, paths) unchanged. "
    "If <KEEP>...</KEEP> tags appear, copy their contents verbatim and remove the tags in your answer. "
    "Return output EXACTLY between <OUT> and </OUT> and nothing else."
)



DOMAIN_TERMS = {
    # finance/reg compliance acronyms & in-house codes you likely see
    "AML","KYC","SAR","SOX","CECL","Basel","GDPR","GLBA","PCI","PII",
    "DMS","GSD","RCC","MRA","MDT","SIFMOS","Minerva","ADM100","MRFXX"
}

FILE_EXTS = ("pdf","doc","docx","xls","xlsx","csv","ppt","pptx","txt")

_DASH = r"[-\u2013\u2014]"  # hyphen-minus, en dash, em dash

def _wrap_keep(s: str, span: Tuple[int, int]) -> str:
    return s[:span[0]] + "<KEEP>" + s[span[0]:span[1]] + "</KEEP>" + s[span[1]:]

def _already_kept(s: str, start: int, end: int) -> bool:
    return "<KEEP>" in s[start:end] or "</KEEP>" in s[start:end]

def _lock_entities(text: str) -> str:
    s = text

    # 0) File names and simple paths (do these first; they can contain dashes)
    file_pat = rf"\b[\w\-. ]+\.({'|'.join(FILE_EXTS)})\b"
    path_pat = r"(?:(?:[A-Za-z]:\\|\/)[\w\-. \/\\]+)"
    for pat in [file_pat, path_pat]:
        for m in list(re.finditer(pat, s)):
            if not _already_kept(s, m.start(), m.end()):
                s = _wrap_keep(s, (m.start(), m.end()))

    # 1) Hyphen/ndash/emdash chains (lock the whole slug)
    hyphen_chain = rf"\b[0-9A-Za-z_]+(?:{_DASH}[0-9A-Za-z_]+)+\b"
    for m in list(re.finditer(hyphen_chain, s)):
        if not _already_kept(s, m.start(), m.end()):
            s = _wrap_keep(s, (m.start(), m.end()))

    # 2) Explicit domain terms (that aren't already inside KEEP)
    for term in sorted(DOMAIN_TERMS, key=len, reverse=True):
        pat = rf"\b{re.escape(term)}\b"
        for m in list(re.finditer(pat, s)):
            if not _already_kept(s, m.start(), m.end()):
                s = _wrap_keep(s, (m.start(), m.end()))

    # 3) IDs, dates, amounts
    patterns = [
        r"\b[A-Z]{2,}\d{2,}\b",          # RCC2024
        r"\b[A-Z]+-\d+\b",               # CASE-10438
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b(?:\$|USD)\s?\d[\d,]*(?:\.\d+)?\b",
        r"\b\d{4,}\b",
    ]
    for pat in patterns:
        for m in list(re.finditer(pat, s)):
            if not _already_kept(s, m.start(), m.end()):
                s = _wrap_keep(s, (m.start(), m.end()))

    # 4) Broad safety net for leftover pure numbers
    for m in list(re.finditer(r"\b\d[\d,./:-]*\b", s)):
        if not _already_kept(s, m.start(), m.end()):
            s = _wrap_keep(s, (m.start(), m.end()))

    # 5) Merge adjacent KEEP chunks joined by a dash into a single KEEP
    #    e.g., <KEEP>open</KEEP>-<KEEP>AML</KEEP>-<KEEP>KYC</KEEP> -> <KEEP>open-AML-KYC</KEEP>
    # merge_pat = rf"(?:<KEEP>[^<]+</KEEP>)(?:{_DASH}(?:<KEEP>[^<]+</KEEP>))+"
    merge_pat = rf"(?:<KEEP>[^<]*?</KEEP>)(?:{_DASH}(?:<KEEP>[^<]*?</KEEP>))+"
    def _merge_span(m):
        raw = m.group(0)
        # strip inner KEEP tags and re-wrap the whole chain
        inner = re.sub(r"</?KEEP>", "", raw)
        return f"<KEEP>{inner}</KEEP>"
    s = re.sub(merge_pat, _merge_span, s)

    return s




# def _wrap_keep(text, span):
#     return text[:span[0]] + "<KEEP>" + text[span[0]:span[1]] + "</KEEP>" + text[span[1]:]

# def _lock_entities(text: str) -> str:
#     """
#     Tag tokens we must not alter:
#       - Domain acronyms/codes (AML, KYC, ADM100, MRFXX, etc.)
#       - Uppercase acronyms with digits (e.g., RCC2024)
#       - Case IDs / ticket-like IDs (ABC-12345)
#       - File names with known extensions
#       - Absolute/relative paths
#       - Dates, times, currency/amounts, plain numbers
#     """
#     s = text

#     # 1) Explicit domain terms
#     for term in sorted(DOMAIN_TERMS, key=len, reverse=True):
#         s = re.sub(rf"\b{re.escape(term)}\b", lambda m: f"<KEEP>{m.group(0)}</KEEP>", s)
    
#     s = re.sub(r"\b\w+(?:-\w+)+\b", lambda m: f"<KEEP>{m.group(0)}</KEEP>", s)
#     # 2) Acronym+digits and ticket-style IDs
#     patterns = [
#         r"\b[A-Z]{2,}\d{2,}\b",          # e.g., RCC2024
#         r"\b[A-Z]+-\d+\b",               # e.g., CASE-12345
#         r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",        # dates 2024-07-31 / 2024/07/31
#         r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",      # dates 07/31/2024
#         r"\b(?:\$|USD)\s?\d[\d,]*(?:\.\d+)?\b",    # amounts $1,200.50 / USD 5000
#         r"\b\d{4,}\b",                              # long numbers (ids, tickets)
#     ]

#     # 3) File names and simple paths
#     file_pat = rf"\b[\w\-. ]+\.({'|'.join(FILE_EXTS)})\b"
#     path_pat = r"(?:(?:[A-Za-z]:\\|\/)[\w\-. \/\\]+)"

#     for pat in [file_pat, path_pat] + patterns:
#         for m in list(re.finditer(pat, s)):
#             # avoid double-wrapping
#             if "<KEEP>" in s[m.start():m.end()]:
#                 continue
#             s = _wrap_keep(s, (m.start(), m.end()))

#     # 4) Numbers/dates/codes not already wrapped (broad safety net)
#     def tag(m): return f"<KEEP>{m.group(0)}</KEEP>"
#     s = re.sub(r"\b\d[\d,./:-]*\b", tag, s)

#     return s

def _contains_negation(s: str) -> bool:
    toks = re.findall(r"[A-Za-z]+'t|[A-Za-z]+|[0-9]+", s.lower())
    return any(tok in NEG_TOKENS for tok in toks)

class Llama3Negator:
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        nli_model: str = "roberta-base-mnli",   # use roberta-large-mnli if you have more VRAM
        dtype = None,
    ):
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if dtype is None:
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
        ).eval()

        self.nli = pipeline(
            "text-classification",
            model=nli_model,
            tokenizer=nli_model,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
        )

    def _build_inputs(self, sentences: List[str]):
        msgs_batch = []
        for s in sentences:
            locked = _lock_entities(s)
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                # {"role": "user", "content": f"Sentence: {locked}\nNegated:"},
                # In _build_inputs, change the user content to explicitly request the tag-wrapped output:
                {"role": "user", "content": f"Sentence: {locked}\nReturn ONLY:\n<OUT>negated sentence here</OUT>"}  
            ]
            msgs_batch.append(msgs)

        inputs = self.tok.apply_chat_template(
            msgs_batch, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True
        ).to(self.model.device)
        return inputs

    @torch.inference_mode()
    def _generate(
        self,
        sentences: List[str],
        n_candidates: int = 6,
        style: str = "beam",                 # "beam" (faithful) or "sample" (diverse)
        max_new_tokens: int = 64,
    ) -> List[List[str]]:
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"  # ensure padding is on the left for Llama3
        inputs = self._build_inputs(sentences)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_candidates,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
            early_stopping=True,
        )
        if style == "beam":
            gen_kwargs.update(dict(do_sample=False, num_beams=max(4, n_candidates)))
        else:
            gen_kwargs.update(dict(do_sample=True, temperature=0.9, top_p=0.92))

        stops = StoppingCriteriaList([StopOnOutClose(self.tok)])
        out = self.model.generate(inputs, stopping_criteria=stops, **gen_kwargs)




        # out = self.model.generate(inputs, **gen_kwargs)
        # Keep only tokens after the prompt to avoid echo
        gen_only = out[:, inputs.shape[1]:]
        dec = self.tok.batch_decode(gen_only, skip_special_tokens=True)

        grouped = [dec[i:i+n_candidates] for i in range(0, len(dec), n_candidates)]
        cleaned: List[List[str]] = []
        for src, cands in zip(sentences, grouped):
            keep, seen = [], set()
            for c in cands:
                payload = _extract_out(c)
                payload = re.sub(r"</?KEEP>", "", payload).strip()
                if not payload or payload.lower() == src.strip().lower():
                    continue
                if payload in seen:
                    continue
                seen.add(payload)
                keep.append(payload)
            cleaned.append(keep or [re.sub(r"</?KEEP>", "", _extract_out(cands[0])).strip()])
        return cleaned




        # grouped = [dec[i:i+n_candidates] for i in range(0, len(dec), n_candidates)]
        # cleaned: List[List[str]] = []
        # for src, cands in zip(sentences, grouped):
        #     keep, seen = [], set()
        #     for c in cands:
        #         s = re.sub(r"</?KEEP>", "", c).strip()
        #         if not s or s.lower() == src.strip().lower():
        #             continue
        #         if s in seen:
        #             continue
        #         seen.add(s)
        #         keep.append(s)
        #     cleaned.append(keep or cands)
        # return cleaned

    def _nli_contradiction_scores(self, pairs: List[Tuple[str, str]]) -> List[float]:
        # MNLI expects premise=hypothesis pairs; we want CONTRADICTION score
        inputs = [{"text": p, "text_pair": h} for (p, h) in pairs]
        results = self.nli(inputs, batch_size=16)
        scores = []
        for r in results:
            contr = next((x for x in r if x["label"].upper().endswith("CONTRADICTION")), None)
            scores.append(float(contr["score"]) if contr else 0.0)
        return scores

    def negate(
        self,
        texts,
        n_candidates: int = 6,
        style: str = "beam",
        negation_mode: str = "explicit",      # "explicit" | "implicit" | "either"
        avoid_double_negation: bool = True,
        min_contradiction: float = 0.5,       # NLI threshold
    ):
        """
        Negate each input using Llama-3 and select the best candidate via MNLI contradiction.

        negation_mode:
        - "explicit": require explicit neg tokens ("not", "n't", "never", etc.).
        - "implicit": require implicit negation patterns ("fail to", "without", "no longer", ...), and
                        reject candidates that contain explicit neg tokens.
        - "either":   accept either explicit or implicit negation forms.

        avoid_double_negation:
        - If True, filter out candidates that contain 2+ neg markers (e.g., "do not fail to ...").
        """
        single = False
        if isinstance(texts, str):
            texts, single = [texts], True

        cand_lists = self._generate(texts, n_candidates=n_candidates, style=style)

        # Collect pairs for NLI scoring after filtering by negation type / double negation
        pairs, idx_map = [], []
        for i, (src, cands) in enumerate(zip(texts, cand_lists)):
            for j, c in enumerate(cands):
                c_stripped = c.strip()
                if not c_stripped:
                    continue

                has_exp = _contains_explicit_negation(c_stripped)
                has_imp = _contains_implicit_negation(c_stripped)
                if avoid_double_negation and _is_double_negation(c_stripped):
                    continue

                # Mode-specific gating
                if negation_mode == "explicit" and not has_exp:
                    continue
                if negation_mode == "implicit" and (not has_imp or has_exp):
                    continue
                if negation_mode == "either" and not (has_exp or has_imp):
                    continue

                pairs.append((src, c_stripped))
                idx_map.append((i, j))

        # If everything got filtered, gracefully fall back to 'either' and allow double-neg once
        relaxed = False
        if not pairs:
            for i, (src, cands) in enumerate(zip(texts, cand_lists)):
                for j, c in enumerate(cands):
                    c_stripped = c.strip()
                    if not c_stripped:
                        continue
                    pairs.append((src, c_stripped))
                    idx_map.append((i, j))
            relaxed = True

        scores = self._nli_contradiction_scores(pairs) if pairs else []

        # Select top candidate per input by contradiction score (subject to threshold)
        best = [""] * len(texts)
        best_score = [-1.0] * len(texts)
        for (i, j), s in zip(idx_map, scores):
            if s >= min_contradiction and s > best_score[i]:
                best[i], best_score[i] = cand_lists[i][j], s

        # Fallbacks: highest-scored for that input, else first candidate
        for i in range(len(texts)):
            if not best[i]:
                indices = [k for k, (ii, _) in enumerate(idx_map) if ii == i]
                if indices:
                    k_best = max(indices, key=lambda k: scores[k])
                    _, j = idx_map[k_best]
                    best[i] = cand_lists[i][j]
                else:
                    best[i] = cand_lists[i][0] if cand_lists[i] else ""

        return best[0:1] if single else best



# ---------- Demo ----------
# pip install -U transformers torch accelerate bitsandbytes
# (MNLI verifier)
# pip install -U transformers

# from your_module import Llama3Negator   # if the class is in another file
# (Assumes Llama3Negator class from earlier is already defined in this script.)

if __name__ == "__main__":
    negator = Llama3Negator(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        nli_model="roberta-large-mnli",   # use "roberta-large-mnli" if you have more VRAM
        # load_in_8bit=True,               # set False if you don't want 8-bit on GPU
    )

    # Short, query-style inputs (financial/internal-doc search flavor)
    queries = [
        "download-SAR-filings",
        "open-AML-KYC-file-2023",
        "retrieve RCC mapping spreadsheet.xlsx",
        "export DMS inventory report",
        "view ADM100 guideline",
        "access GSD folder 2024-07-15",
        "share PCI logs",
        "override retention policy",
        "delete audit trail",
        "access SAR case CASE-10438",
    ]

    def run_case(title, **kwargs):
        print(f"\n=== {title} ===")
        outs = negator.negate(
            queries,
            n_candidates=6,         # generate 6 candidates per input
            style="beam",           # "beam" for faithful; "sample" for diversity
            min_contradiction=0.65, # raise for stricter negations (e.g., 0.7–0.8)
            **kwargs
        )
        for q, n in zip(queries, outs):
            print(f"- {q} -> {n}")

    # 1) Explicit negation (like “not / don't / isn't …”)
    run_case("Explicit negation", negation_mode="explicit", avoid_double_negation=True)

    # 2) Implicit negation only (e.g., “refuse to …”, “decline to …”, “no longer …”)
    #    Explicit “not/n't” forms will be rejected.
    run_case("Implicit negation only", negation_mode="implicit", avoid_double_negation=True)

    # 3) Either explicit or implicit (pick whichever best contradicts the source)
    run_case("Either explicit or implicit", negation_mode="either", avoid_double_negation=True)

    # 4) Diverse variants (sampling) — useful if you want style variety, then post-select
    outs_diverse = negator.negate(
        queries,
        n_candidates=8,
        style="sample",               # temperature/top_p inside the class
        negation_mode="either",
        avoid_double_negation=True,
        min_contradiction=0.7
    )
    # print("\n=== Diverse (sampling) ===")
    # for q, n in zip(queries, outs_diverse):
    #     print(f"- {q} -> {n}")

    # 5) Single-string usage (returns a single-element list; take [0])
    print("\n=== Single input negation ===")
    query = "decrease credit limit"
    single = negator.negate("decrease credit limit", negation_mode="explicit", avoid_double_negation=False)
    print("\nSingle input:")
    print(f"{query} ->", single[0])












