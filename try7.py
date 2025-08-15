# pip install -U transformers torch accelerate bitsandbytes
# (optional but recommended for selection) pip install -U sentencepiece
# Verifier: pip install -U transformers torch
# You must have access to: meta-llama/Meta-Llama-3-8B-Instruct (HF gated model)
# If needed: huggingface-cli login

import re
from typing import List, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NEG_TOKENS = {
    "not", "n't", "never", "no", "cannot", "can't", "won't", "doesn't",
    "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't",
    "couldn't", "mustn't", "mightn't", "shan't"
}



# --- Replace your existing SYSTEM_PROMPT and _lock_entities, and the demo block ---



SYSTEM_PROMPT = (
    "You are a precise editor. Negate the user's short sentence or query with the FEWEST possible edits. "
    "Treat keyword-like inputs (e.g., 'download SAR filings') as imperatives and output a grammatical negated form "
    "('Do not download SAR filings.'). Preserve tense/person and keep ALL entities (names, acronyms, dates, numbers, "
    "codes, file names, paths) unchanged. If <KEEP>...</KEEP> tags appear, copy their contents verbatim and remove "
    "the tags in your answer. Output ONLY the negated sentence, nothing else."
)

DOMAIN_TERMS = {
    # finance/reg compliance acronyms & in-house codes you likely see
    "AML","KYC","SAR","SOX","CECL","Basel","GDPR","GLBA","PCI","PII",
    "DMS","GSD","RCC","MRA","MDT","SIFMOS","Minerva","ADM100","MRFXX"
}

FILE_EXTS = ("pdf","doc","docx","xls","xlsx","csv","ppt","pptx","txt")

def _wrap_keep(text, span):
    return text[:span[0]] + "<KEEP>" + text[span[0]:span[1]] + "</KEEP>" + text[span[1]:]

def _lock_entities(text: str) -> str:
    """
    Tag tokens we must not alter:
      - Domain acronyms/codes (AML, KYC, ADM100, MRFXX, etc.)
      - Uppercase acronyms with digits (e.g., RCC2024)
      - Case IDs / ticket-like IDs (ABC-12345)
      - File names with known extensions
      - Absolute/relative paths
      - Dates, times, currency/amounts, plain numbers
    """
    s = text

    # 1) Explicit domain terms
    for term in sorted(DOMAIN_TERMS, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(term)}\b", lambda m: f"<KEEP>{m.group(0)}</KEEP>", s)

    # 2) Acronym+digits and ticket-style IDs
    patterns = [
        r"\b[A-Z]{2,}\d{2,}\b",          # e.g., RCC2024
        r"\b[A-Z]+-\d+\b",               # e.g., CASE-12345
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",        # dates 2024-07-31 / 2024/07/31
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",      # dates 07/31/2024
        r"\b(?:\$|USD)\s?\d[\d,]*(?:\.\d+)?\b",    # amounts $1,200.50 / USD 5000
        r"\b\d{4,}\b",                              # long numbers (ids, tickets)
    ]

    # 3) File names and simple paths
    file_pat = rf"\b[\w\-. ]+\.({'|'.join(FILE_EXTS)})\b"
    path_pat = r"(?:(?:[A-Za-z]:\\|\/)[\w\-. \/\\]+)"

    for pat in [file_pat, path_pat] + patterns:
        for m in list(re.finditer(pat, s)):
            # avoid double-wrapping
            if "<KEEP>" in s[m.start():m.end()]:
                continue
            s = _wrap_keep(s, (m.start(), m.end()))

    # 4) Numbers/dates/codes not already wrapped (broad safety net)
    def tag(m): return f"<KEEP>{m.group(0)}</KEEP>"
    s = re.sub(r"\b\d[\d,./:-]*\b", tag, s)

    return s

# ---------------- Demo with internal-document search queries ----------------
if __name__ == "__main__":
    negator = Llama3Negator(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        nli_model="roberta-base-mnli",
        load_in_8bit=True,
    )

    examples = [
        # internal doc / compliance / records searches (short queries)
        "download SAR filings",
        "open AML KYC file 2023",
        "retrieve RCC mapping spreadsheet.xlsx",
        "export DMS inventory report",
        "view ADM100 guideline",
        "access GSD folder 2024-07-15",
        "email customer PII",
        "share PCI logs",
        "override retention policy",
        "delete audit trail",
        "bypass KYC checklist",
        "access SAR case CASE-10438",
        "fetch MRFXX validation report",
        "publish MRA closeout memo",
        "copy Basel III workbook.xlsx to /restricted/reports/",
    ]

    out = negator.negate(
        examples,
        n_candidates=6,
        style="beam",
        ensure_neg_token=True,
        min_contradiction=0.65
    )

    for q, neg in zip(examples, out):
        print(f"{q}  ->  {neg}")
