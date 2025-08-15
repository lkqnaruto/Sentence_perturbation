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
                {"role": "user", "content": f"Sentence: {locked}\nNegated:"},
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

        out = self.model.generate(inputs, **gen_kwargs)
        # Keep only tokens after the prompt to avoid echo
        gen_only = out[:, inputs.shape[1]:]
        dec = self.tok.batch_decode(gen_only, skip_special_tokens=True)

        grouped = [dec[i:i+n_candidates] for i in range(0, len(dec), n_candidates)]
        cleaned: List[List[str]] = []
        for src, cands in zip(sentences, grouped):
            keep, seen = [], set()
            for c in cands:
                s = re.sub(r"</?KEEP>", "", c).strip()
                if not s or s.lower() == src.strip().lower():
                    continue
                if s in seen:
                    continue
                seen.add(s)
                keep.append(s)
            cleaned.append(keep or cands)
        return cleaned

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
        texts: Union[str, List[str]],
        n_candidates: int = 6,
        style: str = "beam",
        ensure_neg_token: bool = False,       # require explicit negation tokens (“not”, “isn't”, …)
        min_contradiction: float = 0.5,      # NLI threshold
    ) -> List[str]:
        single = False
        if isinstance(texts, str):
            texts, single = [texts], True

        cand_lists = self._generate(texts, n_candidates=n_candidates, style=style)

        pairs, idx_map = [], []
        for i, (src, cands) in enumerate(zip(texts, cand_lists)):
            for j, c in enumerate(cands):
                if ensure_neg_token and not _contains_negation(c):
                    continue
                pairs.append((src, c))
                idx_map.append((i, j))

        # If everything got filtered by ensure_neg_token, relax it once
        relaxed = False
        if not pairs:
            for i, (src, cands) in enumerate(zip(texts, cand_lists)):
                for j, c in enumerate(cands):
                    pairs.append((src, c)); idx_map.append((i, j))
            relaxed = True

        scores = self._nli_contradiction_scores(pairs) if pairs else []

        best = [""] * len(texts)
        best_score = [-1.0] * len(texts)
        for (i, j), s in zip(idx_map, scores):
            if s >= min_contradiction and s > best_score[i]:
                best[i], best_score[i] = cand_lists[i][j], s

        # Fallbacks: top-scoring for that input, else first candidate
        for i in range(len(texts)):
            if not best[i]:
                indices = [k for k, (ii, jj) in enumerate(idx_map) if ii == i]
                if indices:
                    k_best = max(indices, key=lambda k: scores[k])
                    _, j = idx_map[k_best]
                    best[i] = cand_lists[i][j]
                else:
                    best[i] = cand_lists[i][0] if cand_lists[i] else ""

        return best[0:1] if single else best


# ---------- Demo ----------
if __name__ == "__main__":
    negator = Llama3Negator(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        nli_model="roberta-large-mnli"
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
        ensure_neg_token=False,
        min_contradiction=0.65
    )

    for q, neg in zip(examples, out):
        print(f"{q}  ->  {neg}")
