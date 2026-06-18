# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Build a ~target-token needle-in-a-haystack prompt for the iOS Gemma 4 131k test.

A unique fact (the "needle") is woven into varied filler prose at ~12% depth, then
a question is asked at the very end. Sized with the real Gemma 4 tokenizer to land
just under a target token count (default 115000, safely below the 131072 cap).

    uv run --with "transformers>=5.5.0" python python/scripts/build_needle_prompt.py \
        --out /tmp/needle_120k.txt --target-tokens 115000
"""

import argparse

from transformers import AutoTokenizer

NEEDLE = (
    "Important fact to remember: the secret access code for the Aurora research "
    "vault is 7741-MAGENTA-9. Keep this code in mind."
)

QUESTION = (
    "\n\nNow, answer this question based only on the text above. "
    "What is the secret access code for the Aurora research vault? "
    "Answer with just the code.\nAnswer:"
)

# Varied filler so the haystack isn't a degenerate repeat (which can confuse recall).
FILLER_SENTENCES = [
    "The committee reviewed the quarterly logistics report before lunch.",
    "Rainfall across the northern provinces remained well below the seasonal average.",
    "A new species of beetle was catalogued near the river delta last spring.",
    "The orchestra rehearsed the second movement until the acoustics felt right.",
    "Supply chains for rare-earth magnets continued to tighten through the year.",
    "Volunteers repainted the old lighthouse a brilliant shade of white.",
    "The lecture covered the thermodynamics of shallow coastal currents.",
    "Market analysts disagreed about the direction of the bond yields.",
    "Migratory cranes paused at the wetland reserve on their way south.",
    "The bakery introduced a sourdough loaf made with ancient grains.",
    "Engineers tested the bridge expansion joints under simulated traffic.",
    "A documentary about deep-sea vents premiered to quiet acclaim.",
    "The archive digitized thousands of fragile nineteenth-century letters.",
    "Snowfall in the high passes closed the road for nearly a week.",
    "Researchers mapped the genome of a drought-resistant cactus.",
    "The festival featured kites shaped like dragons and silver fish.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-tokens", type=int, default=115000)
    ap.add_argument("--needle-depth", type=float, default=0.12)
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    intro = (
        "You are reading a long document. Read carefully; a question follows at the end.\n\n"
    )

    # Build filler until we're close to the target, inserting the needle at depth.
    body_parts = []
    needle_inserted = False
    i = 0
    # Reserve room for the question + chat template overhead.
    budget = args.target_tokens - 256

    def n_tokens(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    cur_tokens = n_tokens(intro)
    while cur_tokens < budget:
        if not needle_inserted and cur_tokens >= budget * args.needle_depth:
            body_parts.append(" " + NEEDLE + " ")
            cur_tokens += n_tokens(NEEDLE)
            needle_inserted = True
        # Add a small block of varied filler at a time.
        block = " ".join(
            FILLER_SENTENCES[(i + k) % len(FILLER_SENTENCES)] for k in range(8)
        )
        body_parts.append(" " + block)
        cur_tokens += n_tokens(block)
        i += 1
    if not needle_inserted:
        body_parts.insert(1, " " + NEEDLE + " ")

    prompt = intro + "".join(body_parts) + QUESTION
    total = n_tokens(prompt)
    with open(args.out, "w") as f:
        f.write(prompt)
    print(f"wrote {args.out}")
    print(f"approx tokens (no chat template): {total}")
    print(f"needle inserted at ~{args.needle_depth*100:.0f}% depth; needle code = 7741-MAGENTA-9")


if __name__ == "__main__":
    main()
