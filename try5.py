# pip install -U transformers torch
# Optional (for the verifier): pip install -U accelerate
# Optional (for faster CPU/GPU batching): pip install -U einops

from typing import List, Union, Tuple
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# from transformers import pipeline
# nli = pipeline("text-classification", model="roberta-large-mnli")



NEG_LEXEMES = {
    "not","n't","never","no","none","nobody","nowhere","nothing",
    "cannot","can't","won't","doesn't","don't","didn't","isn't","aren't","wasn't","weren't",
    "hasn't","haven't","hadn't","shouldn't","wouldn't","couldn't","mustn't","mightn't","shan't"
}

class TransformerNegator:
    """
    Generate a negated version of a sentence using a seq2seq transformer (FLAN-T5 by default),
    then select the candidate that most strongly contradicts the original using an MNLI verifier.
    """
    def __init__(
        self,
        generator_id: str = "google/flan-t5-base",     # swap to "t5-base" or "t5-small" if needed
        verifier_id: str = "roberta-base-mnli",        # MNLI contradiction checker
    ):
        # Generator (seq2seq)
        self.gen_tok = AutoTokenizer.from_pretrained(generator_id, use_fast=True)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(generator_id).to(DEVICE).eval()

        # Verifier (NLI pipeline)
        self.nli = pipeline(
            "text-classification",
            model=verifier_id,
            tokenizer=verifier_id,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1,
        )

    def _prep_prompts(self, texts: List[str]) -> List[str]:
        # Instruction tuned for faithful, minimal edits and entity preservation.
        prompts = []
        for t in texts:
            prompts.append(
                "Negate the sentence while preserving tense, person, and named entities. "
                "Use minimal edits and correct grammar. Output only the negated sentence.\n\n"
                f"Sentence: {t}\nNegated:"
            )
        return prompts

    @torch.inference_mode()
    def _generate_candidates(
        self,
        texts: List[str],
        n: int = 6,
        style: str = "beam",            # "beam" (faithful) or "sample" (diverse)
        max_new_tokens: int = 64,
    ) -> List[List[str]]:
        prompts = self._prep_prompts(texts)
        enc = self.gen_tok(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(DEVICE)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=True,
        )
        if style == "beam":
            gen_kwargs.update(dict(num_beams=max(4, n), do_sample=False))
        else:
            gen_kwargs.update(dict(do_sample=True, temperature=0.9, top_p=0.92))

        out = self.gen_model.generate(**enc, **gen_kwargs)
        decoded = self.gen_tok.batch_decode(out, skip_special_tokens=True)

        # Group per input and light cleanup
        grouped = [decoded[i:i+n] for i in range(0, len(decoded), n)]
        cleaned: List[List[str]] = []
        for orig, cands in zip(texts, grouped):
            keep, seen = [], set()
            for c in cands:
                s = c.strip()
                if not s or s.lower() == orig.strip().lower():
                    continue
                # drop exact dupes
                if s in seen:
                    continue
                seen.add(s)
                keep.append(s)
            cleaned.append(keep or cands)
        return cleaned

    def _contains_negation(self, s: str) -> bool:
        toks = re.findall(r"[A-Za-z]+'t|[A-Za-z]+|[0-9]+", s.lower())
        return any(tok in NEG_LEXEMES for tok in toks)

    def _nli_contradiction_scores(
        self, pairs: List[Tuple[str, str]]
    ) -> List[float]:
        # Batch MNLI inference; extract contradiction score
        inputs = [{"text": p, "text_pair": h} for (p, h) in pairs]
        results = self.nli(inputs, batch_size=16)
        scores = []
        for r in results:
            # labels: CONTRADICTION / NEUTRAL / ENTAILMENT
            contr = next((x for x in r if x["label"].upper().endswith("CONTRADICTION")), None)
            scores.append(float(contr["score"]) if contr else 0.0)
        return scores

    def negate(
        self,
        texts: Union[str, List[str]],
        n_candidates: int = 6,
        style: str = "beam",
        ensure_neg_token: bool = True,     # enforce presence of typical negation words
        min_contradiction: float = 0.50,   # filter by NLI contradiction prob
    ) -> List[str]:
        """Return one negated sentence per input (best candidate by contradiction score)."""
        single = False
        if isinstance(texts, str):
            texts, single = [texts], True

        cand_lists = self._generate_candidates(texts, n=n_candidates, style=style)

        # Flatten pairs for one-shot NLI scoring
        pairs, idx_map = [], []
        for i, (src, cands) in enumerate(zip(texts, cand_lists)):
            for j, c in enumerate(cands):
                if ensure_neg_token and not self._contains_negation(c):
                    continue
                pairs.append((src, c))
                idx_map.append((i, j))

        # If everything was filtered out by ensure_neg_token, relax that constraint once
        relaxed = False
        if not pairs:
            for i, (src, cands) in enumerate(zip(texts, cand_lists)):
                for j, c in enumerate(cands):
                    pairs.append((src, c))
                    idx_map.append((i, j))
            relaxed = True

        scores = self._nli_contradiction_scores(pairs) if pairs else []

        # Pick best per input
        best = [""] * len(texts)
        best_score = [-1.0] * len(texts)
        for (i, j), s in zip(idx_map, scores):
            if s >= min_contradiction and s > best_score[i]:
                best[i], best_score[i] = cand_lists[i][j], s

        # Fallbacks (no candidate passed threshold): take top-scoring overall or first candidate
        for i in range(len(texts)):
            if not best[i]:
                # pick highest score for that input if available, otherwise first candidate
                # find scores for input i
                indices = [k for k, (ii, jj) in enumerate(idx_map) if ii == i]
                if indices:
                    k_best = max(indices, key=lambda k: scores[k])
                    _, j = idx_map[k_best]
                    best[i] = cand_lists[i][j]
                else:
                    best[i] = cand_lists[i][0] if cand_lists[i] else ""

        # Optional: annotate if we had to relax token constraint
        if relaxed:
            pass  # you could log this condition for audits

        return best[0:1] if single else best


# ----------------- Demo -----------------
if __name__ == "__main__":
    negator = TransformerNegator(
        generator_id="google/flan-t5-base",   # try "t5-small" for CPU, or "google/flan-t5-large" if you have GPU
        verifier_id="roberta-large-mnli",      # can upgrade to roberta-large-mnli if you have VRAM
    )

    examples = [
        "model life cycle procedure",
        "He goes to the office.",
        "We have finished the report.",
        "She not happy.",
        "Close the door.",
        "I will submit the validation memo tomorrow.",
        "The model underperforms on long sentences.",
    ]
    negated = negator.negate(examples, n_candidates=6, style="beam",
                             ensure_neg_token=True, min_contradiction=0.6)
    for src, tgt in zip(examples, negated):
        print(f"{src}  -->  {tgt}")
