# pip install "transformers>=4.41" accelerate torch bitsandbytes --upgrade
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,

)

SYSTEM_PROMPT = (
    "You are a precise paraphraser. Rewrite the user's text to preserve meaning, "
    "tone, and factual details. Do not introduce new facts. Keep numbers, dates, "
    "codes, and names unchanged. If <KEEP>...</KEEP> tags appear, copy the text "
    "inside them verbatim and remove the tags in your final answer. Output only the paraphrase."
)

def _lock_entities(txt: str):
    # Tag numbers/dates/ids to keep them unchanged (simple heuristic).
    def tag(m): return f"<KEEP>{m.group(0)}</KEEP>"
    return re.sub(r"\b\d[\d,.:/\-]*\b", tag, txt)

@torch.inference_mode()
def paraphrase_llama3(
    text: str,
    n: int = 3,
    style: str = "conservative",   # "conservative" (beam) or "diverse" (sampling)
    max_new_tokens: int = 32,
):
    locked = _lock_entities(text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Paraphrase this:\n\n{locked}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=3,
        repetition_penalty=1.1,
        num_return_sequences=n,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    if style == "conservative":
        gen_kwargs.update(dict(do_sample=False, num_beams=max(4, n), early_stopping=True))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=0.9, top_p=0.92))

    out = model.generate(inputs, **gen_kwargs)

    # Keep only generated tokens after the prompt to avoid echo
    gen_only = out[:, inputs.shape[1]:]
    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    # Clean up tags & dedup
    cleaned = []
    for s in decoded:
        s = re.sub(r"</?KEEP>", "", s).strip()
        cleaned.append(s)

    uniq, seen = [], set()
    for s in cleaned:
        if s and s.lower() != text.strip().lower() and s not in seen:
            seen.add(s); uniq.append(s)
    return uniq or cleaned

if __name__ == "__main__":
    src = "We will finalize the validation report by August 15, 2025 and share it with the audit team."
    print("— Conservative —")
    for i, p in enumerate(paraphrase_llama3(src, n=3, style="conservative"), 1):
        print(f"{i}. {p}")
    print("\n— Diverse —")
    for i, p in enumerate(paraphrase_llama3(src, n=3, style="diverse"), 1):
        print(f"{i}. {p}")
