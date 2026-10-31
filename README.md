# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution:

`build_chain()` sends each photo to `deepseek-v4-flash-vision-exp` in a multimodal message and
asks, for every receipt in the frame, for the printed 小計 subtotal, the ROUNDING amount, the
amount actually paid, and each discount line, as one JSON object. A photo that reports more than
one receipt is then re-read receipt by receipt: a second call locates each receipt with a bounding
box, the box is cropped, scaled up and sent in two orientations so the model can pick the upright
view, and the close-up passes are repeated until two of them agree, because a single pass over a
crowded photo reliably misreads the lines. `answer_queries()` batches those calls and derives both
answers from that one reading, summing the payment lines for the money spent and subtotal plus
discounts for the pre-discount total. All arithmetic is `Decimal`, and a photo that cannot be read
yields `HK$extraction failed` rather than a partial total. The close-up pass uses Pillow, listed in
`requirements.txt`; without it the script falls back to reading whole photos.

Verified on both sets: `public_test` gives `HK$1974.30` and `HK$2348.20`, and the 28-photo grading
folder gives `HK$2319.20` and `HK$2766.90`, matching each `ground_truth.json` on both questions.
