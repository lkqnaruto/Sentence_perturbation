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
    "You are a precise editor. Negate the user's sentence with the FEWEST possible edits. "
    "Preserve tense, person, and all entities (names, dates, numbers, codes). "
    "If <KEEP>...</KEEP> tags appear, copy their contents verbatim and remove the tags in your answer. "
    "Output ONLY the negated sentence, nothing else."
)

def _lock_entities(text: str) -> str:
    # Lightweight locking of numbers/dates/codes; extend to names if needed
    def tag(m): return f"<KEEP>{m.group(0)}</KEEP>"
    text = re.sub(r"\b\d[\d,./:-]*\b", tag, text)  # numbers/dates
    return text

def _contains_negation(s: str) -> bool:
    toks = re.findall(r"[A-Za-z]+'t|[A-Za-z]+|[0-9]+", s.lower())
    return any(tok in NEG_TOKENS for tok in toks)

class Llama3Negator:
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        nli_model: str = "roberta-base-mnli",   # use roberta-large-mnli if you have more VRAM
        load_in_8bit: bool = True,              # set False if you don’t want 8-bit
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
        nli_model="roberta-large-mnli",   # or "roberta-large-mnli" for stronger verification
        load_in_8bit=True,
    )

    examples = [
        "reset password",
        "open account",
        "transfer funds",
        "increase credit limit",
        "close credit card",
        "enable two-factor authentication",
        "download statement",
        "update mailing address",
        "schedule payment",
        "verify identity"
    ]
    out = negator.negate(examples, n_candidates=6, style="beam",
                         ensure_neg_token=True, min_contradiction=0.6)
    for s, t in zip(examples, out):
        print(f"{s}  ->  {t}")
