# pip install transformers torch --upgrade
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# You can swap to a small paraphrase-tuned checkpoint (same T5-small size) for stronger results:
# CHECKPOINT = "ramsrigouthamg/t5_paraphraser"
CHECKPOINT = "prithivida/parrot_paraphraser_on_T5"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(DEVICE)
model.eval()

def _prepare_inputs(texts, max_input_len=256):
    if isinstance(texts, str):
        texts = [texts]
    # T5-style task prefix; works best if you use a paraphrase-tuned checkpoint
    prefixed = [f"paraphrase: {t}" for t in texts]
    return tokenizer(
        prefixed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len
    ).to(DEVICE), texts

@torch.inference_mode()
def paraphrase(
    texts,
    n=3,
    strategy="beam",           # "beam" or "sample"
    max_new_tokens=64,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
    temperature=0.9,
    top_p=0.92,
):
    """
    Returns a list of lists: one list of n paraphrases per input text.
    """
    encodings, original = _prepare_inputs(texts)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        num_return_sequences=n,
        early_stopping=True,
    )

    if strategy == "beam":
        gen_kwargs.update(num_beams=max(n, 4), do_sample=False)
    else:  # sampling
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)

    outputs = model.generate(**encodings, **gen_kwargs)

    # Decode and regroup n outputs per input
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    grouped = [decoded[i:i+n] for i in range(0, len(decoded), n)]

    # Post-process: strip, deduplicate, and remove verbatim copies
    cleaned = []
    for orig, candidates in zip(original, grouped):
        uniq = []
        seen = set()
        for c in candidates:
            s = c.strip()
            # drop duplicates and exact copies of the original
            if s and s.lower() != orig.strip().lower() and s not in seen:
                seen.add(s)
                uniq.append(s)
        cleaned.append(uniq or candidates)  # fallback if everything deduped away
    return cleaned

if __name__ == "__main__":
    texts = [
        "The model underperforms on long, compound sentences with rare terminology.",
        "We should schedule the validation review early next week."
    ]

    print("— Beam search paraphrases —")
    for i, outs in enumerate(paraphrase(texts, n=3, strategy="beam")):
        print(f"\nInput {i+1}: {texts[i]}")
        for j, o in enumerate(outs, 1):
            print(f"  {j}. {o}")

    print("\n— Sampling paraphrases —")
    for i, outs in enumerate(paraphrase(texts, n=3, strategy="sample")):
        print(f"\nInput {i+1}: {texts[i]}")
        for j, o in enumerate(outs, 1):
            print(f"  {j}. {o}")
