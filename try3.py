# pip install -U transformers torch sentence-transformers  # (SBERT is optional)

from typing import List, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class T5Paraphraser:
    def __init__(
        self,
        model_id: str = "t5-small",          # e.g., "t5-small", "t5-base", "google/flan-t5-base"
        task_prefix: str = "paraphrase: ",   # many T5 paraphrase checkpoints expect this
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(DEVICE)
        self.model.eval()
        self.task_prefix = task_prefix

        # Optional semantic filter (lazy import)
        self._sbert = None

    def _maybe_load_sbert(self):
        if self._sbert is None:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _prep(self, texts: Union[str, List[str]], max_input_len: int = 256):
        if isinstance(texts, str):
            texts = [texts]
        inputs = [self.task_prefix + t for t in texts]
        enc = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        ).to(DEVICE)
        return enc, texts

    @torch.inference_mode()
    def paraphrase(
        self,
        texts: Union[str, List[str]],
        n: int = 3,
        style: str = "beam",                 # "beam" for fidelity, "sample" for diversity
        max_new_tokens: int = 64,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.15,
        temperature: float = 0.9,
        top_p: float = 0.92,
        semantic_filter: bool = False,
        min_cosine: float = 0.88,            # used if semantic_filter=True
    ) -> List[List[str]]:
        enc, originals = self._prep(texts)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            num_return_sequences=n,
            early_stopping=True,
        )
        if style == "beam":
            gen_kwargs.update(num_beams=max(4, n), do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

        outs = self.model.generate(**enc, **gen_kwargs)
        decoded = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
        grouped = [decoded[i:i+n] for i in range(0, len(decoded), n)]

        # Clean up, dedup, remove exact copies of the original
        cleaned: List[List[str]] = []
        for orig, cands in zip(originals, grouped):
            seen, keep = set(), []
            for c in cands:
                s = c.strip()
                if not s:
                    continue
                if s.lower() == orig.strip().lower():
                    continue
                if s in seen:
                    continue
                seen.add(s)
                keep.append(s)
            cleaned.append(keep or cands)

        # Optional semantic similarity filter for faithfulness
        if semantic_filter:
            self._maybe_load_sbert()
            from sentence_transformers import util as sbert_util
            filtered_all: List[List[str]] = []
            for orig, cands in zip(originals, cleaned):
                if not cands:
                    filtered_all.append([])
                    continue
                o_emb = self._sbert.encode([orig], convert_to_tensor=True)
                c_emb = self._sbert.encode(cands, convert_to_tensor=True)
                sims = sbert_util.cos_sim(o_emb, c_emb).cpu().numpy()[0]
                filtered = [c for c, sim in zip(cands, sims) if sim >= min_cosine]
                filtered_all.append(filtered or cands)  # fall back if all filtered out
            cleaned = filtered_all

        return cleaned

# ---- Example usage ----
if __name__ == "__main__":
    paraphraser = T5Paraphraser(model_id="google/flan-t5-base")  # swap to "t5-base" or "google/flan-t5-base"
    texts = [
        "The model underperforms on long, compound sentences with rare terminology.",
        "We should schedule the validation review early next week."
    ]

    print("— Beam (faithful) —")
    results = paraphraser.paraphrase(texts, n=3, style="beam", semantic_filter=True, min_cosine=0.9)
    for src, outs in zip(texts, results):
        print(f"\nInput: {src}")
        for i, o in enumerate(outs, 1):
            print(f"  {i}. {o}")

    print("\n— Sampling (diverse) —")
    results = paraphraser.paraphrase(texts, n=3, style="sample", semantic_filter=False)
    for src, outs in zip(texts, results):
        print(f"\nInput: {src}")
        for i, o in enumerate(outs, 1):
            print(f"  {i}. {o}")
