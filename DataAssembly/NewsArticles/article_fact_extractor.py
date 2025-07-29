from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from openai import OpenAI
import asyncio
import csv
import json
import os
import ast
from typing import Any, Iterable, List
import httpx
import re
import itertools
OLLAMA_SERVERS = ["http://localhost:11434", "http://localhost:11435"]
server_cycle = itertools.cycle(OLLAMA_SERVERS)
"""
async_pipeline.py
=================
Memory-bounded, batched, asynchronous pipeline that extracts fact-level
events from a large CSV of news articles using an LLM.

Key properties
--------------
* **Bounded RAM** – only `MAX_WORKERS` worker coroutines live at once,
  regardless of how many rows the CSV has.
* **GPU safety** – at most `GPU_CONCURRENCY` simultaneous LLM calls.
* **Batched writes** – facts are flushed to disk every `FLUSH_EVERY`
  records by a single writer coroutine.
* **Crash-resumable** – on start-up the script looks at the output file
  and skips all articles already processed.

Run it simply with:

    python async_pipeline.py

All paths and parameters are hard-coded near the bottom of the file.
"""




###############################################################################
# Prompt builder
###############################################################################
def build_llm_prompt(
    *,
    date: str,
    article_text: str,
    focus_tickers: list[str],
) -> str:
    focus = ", ".join(focus_tickers)
    return f"""
    You are an expert financial news analyst.
    Your task is to extract, for each mentioned company, a structured, concise, yet mearningful summary of the relevant story elements from the article. Return your output as a JSON array where each element follows this schema:
    {{
    "date": "<YYYY-MM-DD>",
    "tickers": ["<TICKER_1>", …],
    "raw_text": "<coherent description of the situation involving these tickers>",
    "event_type": "<string>",
    "sentiment": <float>
    }}

    Rules:
    - Each JSON object should group information relevant to a single ticker or a small set of closely linked tickers.
    - DO NOT return any entry unless at least one of the following tickers is mentioned: {focus}
    - `raw_text` should be a coherent summary (1–3 sentences) capturing the core issue or development for the ticker(s).
    - Choose `event_type` from a sensible finite set such as:
    "earnings_announcement", "partnership", "lawsuit",
    "product_launch", "executive_change", "acquisition",
    "bankruptcy", or "other". Use "other" if no better option fits.
    - `sentiment` must be a float between -1 (very negative) and 1 (very positive), representing the tone toward the affected ticker(s).

    Date: {date}
    Article:
    \"\"\"
    {article_text}
    \"\"\"

    Respond with **only** a single JSON array or the format will be deemed invalid!
    """.strip()



###############################################################################
# LLM response validation
###############################################################################
_REQUIRED_FIELDS = {
    "date": str,
    "tickers": list,
    "raw_text": str,
    "event_type": str,
    "sentiment": (float, int),
}

def validate_llm_response(raw: str) -> list[dict[str, Any]]:
    """
    Extracts and validates a list of fact objects from potentially messy LLM output.
    Raises ValueError if no valid JSON list of fact dicts is found.
    """
    # First try a direct parse
    try:
        data = json.loads(raw)
        return _validate_fact_list(data)
    except Exception:
        pass  # We'll fall back to regex scan

    # Look for any JSON array fragments in the text
    array_candidates = re.findall(r"\[\s*{.*?}\s*\]", raw, re.DOTALL)

    for candidate in array_candidates:
        try:
            data = json.loads(candidate)
            return _validate_fact_list(data)
        except Exception:
            continue

    raise ValueError("No valid JSON array of fact objects found in LLM response.")


def _validate_fact_list(data: Any) -> list[dict[str, Any]]:
    """Internal helper to validate list structure and field types."""
    if not isinstance(data, list):
        raise ValueError("Top-level JSON is not a list")

    for i, fact in enumerate(data):
        if not isinstance(fact, dict):
            raise ValueError(f"Fact {i} is not an object")
        for fld, typ in _REQUIRED_FIELDS.items():
            if fld not in fact:
                raise ValueError(f"Fact {i} missing field '{fld}'")
            if not isinstance(fact[fld], typ):
                raise ValueError(
                    f"Fact {i} field '{fld}' wrong type: expected {typ}, got {type(fact[fld])}"
                )
        # ticker list must be all strings
        if not all(isinstance(t, str) for t in fact["tickers"]):
            raise ValueError(f"Fact {i} has non-string ticker entries")

    return data


###############################################################################
# Async LLM placeholder  (replace with real inference)
###############################################################################


async def call_llm_async(prompt: str, model: str) -> str:
    """
    Asynchronous call to an Ollama model, round-robin distributed across GPUs.
    """
    server = next(server_cycle)
    url = f"{server}/api/generate"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        resp.raise_for_status()
        content = resp.json()
        return content["response"].strip()



###############################################################################
# Disk I/O helpers
###############################################################################
_file_lock = asyncio.Lock()  # ensures only one thread writes at a time


def _append_blocking(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


async def flush_rows(rows: list[dict[str, Any]], path: str) -> None:
    async with _file_lock:
        await asyncio.to_thread(_append_blocking, rows, path)


###############################################################################
# Resume helper  (scan output JSONL for highest processed index)
###############################################################################
def _max_index(path: str) -> int:
    if not os.path.exists(path):
        return -1
    max_idx = -1
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                idx = json.loads(line).get("source_article_index", -1)
                if isinstance(idx, int) and idx > max_idx:
                    max_idx = idx
            except json.JSONDecodeError:
                continue
    return max_idx


###############################################################################
# Core pipeline – bounded memory, batched writes
###############################################################################
async def run_pipeline_async(
    dataset: Iterable[dict[str, Any]],
    *,
    output_path: str,
    model:str,
    max_workers: int = 8,
    gpu_concurrency: int = 4,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    flush_every: int = 100,
) -> None:
    """
    Process *dataset* (iterable of article dicts) and append fact records to *output_path*.

    The function returns when every article has been processed or skipped.
    """

    last_done = _max_index(output_path)
    print(f"[PIPE] Resuming after article index {last_done}")

    article_q: asyncio.Queue[Any] = asyncio.Queue(max_workers * 2)
    fact_q: asyncio.Queue[Any] = asyncio.Queue(flush_every * 2)
    gpu_sem = asyncio.Semaphore(gpu_concurrency)

    # ────────────────────────── writer coroutine ──────────────────────────
    async def writer() -> None:
        buffer: list[dict[str, Any]] = []
        flush_count = 0
        print("[WRITER] started")
        while True:
            fact = await fact_q.get()
            if fact is None:                      # sentinel
                break
            buffer.append(fact)
            if len(buffer) >= flush_every:
                await flush_rows(buffer, output_path)
                flush_count += 1
                # print(f"[WRITER] flushed batch #{flush_count} ({len(buffer)} rows)")
                buffer.clear()
            fact_q.task_done()

        if buffer:
            await flush_rows(buffer, output_path)
            # print(f"[WRITER] final flush ({len(buffer)} rows)")
        fact_q.task_done()
        # print("[WRITER] finished")

    # ────────────────────────── worker coroutine ──────────────────────────
    async def worker(worker_id: int) -> None:
        # print(f"[WORKER {worker_id}] started")
        while True:
            item = await article_q.get()
            if item is None:
                article_q.task_done()
                break

            idx, article = item
            article_q.task_done()
            # print(f"[WORKER {worker_id}] dequeued article {idx}")

            if idx <= last_done:
                continue

            # build focus tickers …
            try:

                raw_assoc = article.get("Associated_tickers", "")

                if isinstance(raw_assoc, str):
                    raw_assoc = raw_assoc.strip()
                    try:
                        # If the string looks like "[...]" try to parse it as a Python list
                        associated = ast.literal_eval(raw_assoc) if raw_assoc.startswith("[") else raw_assoc.split(",")
                    except (ValueError, SyntaxError):
                        # literal_eval failed → treat it as plain text
                        associated = raw_assoc.split(",")
                else:
                    # already some other iterable (unlikely for CSV) → just cast to list
                    associated = list(raw_assoc)

                # final clean-up: strip whitespace / quotes and drop empties
                associated = [t.strip(" '\"").upper() for t in associated if t.strip(" '\"")]

                focus = [article["Stock_symbol"].upper(), *associated]

                prompt = build_llm_prompt(
                    date=article["Date"],
                    article_text=article["Article"],
                    focus_tickers=focus,
                )
            except Exception as e:
                print(f"[WORKER {worker_id}] article {idx} crashed: {e}")
                continue

            validated: list[dict[str, Any]] = []
            async with gpu_sem:
                # print(f"[WORKER {worker_id}] calling LLM for article {idx}")
                for attempt in range(1, max_attempts + 1):
                    try:
                        raw = await call_llm_async(prompt=prompt,
                                                   model=model)
                        print(f"[WORKER {worker_id}] LLM returned for article {idx} (attempt {attempt})")
                        validated = validate_llm_response(raw)
                        print(f"Validated return: {validated}")
                        print(f"Focus tickers: {focus}")
                        break
                    except Exception as exc:
                        print(f"[WORKER {worker_id}] attempt {attempt} failed on article {idx}: {type(exc).__name__}: {exc}")
                        print(f"[WORKER {worker_id}] recieved: {raw}")

                        if attempt < max_attempts:
                            await asyncio.sleep(backoff_base * attempt)

            for fact in validated:
                fact["source_article_index"] = idx
                await fact_q.put(fact)
                # print(f"[WORKER {worker_id}] queued fact for article {idx}")

        # print(f"[WORKER {worker_id}] finished")

    # ────────────────────────── reader coroutine ──────────────────────────
    async def reader() -> None:
        print("[READER] started")
        for idx, art in enumerate(dataset):
            if idx % 10_000 == 0:
                print(f"[READER] queued article {idx}")
            await article_q.put((idx, art))
        for _ in range(max_workers):
            await article_q.put(None)
        print("[READER] pushed sentinels + finished")

    # Launch coroutines
    writer_task = asyncio.create_task(writer())
    worker_tasks = [asyncio.create_task(worker(i)) for i in range(max_workers)]
    reader_task = asyncio.create_task(reader())

    await reader_task
    print("[PIPE] reader_task done; waiting article_q.join()")
    await article_q.join()
    print("[PIPE] article_q drained; signalling writer")

    await fact_q.put(None)
    print("[PIPE] sent sentinel to writer; waiting fact_q.join()")
    await fact_q.join()
    print("[PIPE] fact_q drained; waiting writer_task")
    await writer_task

    print("[PIPE] waiting remaining workers")
    await asyncio.gather(*worker_tasks)
    print("[PIPE] Completed all articles")


###############################################################################
# CSV iterator helper
###############################################################################
def iter_csv(path: str) -> Iterable[dict[str, Any]]:
    """
    Stream rows from a CSV as dicts.  The file handle is kept open for the
    life of the iterator (simple pattern suitable for this script).
    """
    f = open(path, newline="", encoding="utf-8")  # kept open until GC
    return csv.DictReader(f)


###############################################################################
# Hard-coded
###############################################################################
def main() -> None:
    """
    Run the pipeline with fixed paths and parameters.
    Edit the constants below if you need different files or settings.
    """

    # ─── hard-coded configuration ─────────────────────────────────────────
    INPUT_CSV       = config.NEWS_CSV_PATH_ASSOCIATED_TICKERS   # path to source CSV
    OUTPUT_JSONL    = config.NEWS_FACTS    # destination JSONL file
    MAX_WORKERS     = 8                     # worker coroutines (CPU-side)
    GPU_CONCURRENCY = 4                     # simultaneous LLM calls
    FLUSH_EVERY     = 100                   # facts per disk flush
    MAX_ATTEMPTS    = 3                     # retries per article on failure
    MODEL_NAME = "llama3.3:70b"  # change to "llama2:70b" on GPU box
    # ──────────────────────────────────────────────────────────────────────

    dataset = iter_csv(INPUT_CSV)
    asyncio.run(
        run_pipeline_async(
            dataset,
            output_path     = OUTPUT_JSONL,
            model           = MODEL_NAME,
            max_workers     = MAX_WORKERS,
            gpu_concurrency = GPU_CONCURRENCY,
            flush_every     = FLUSH_EVERY,
            max_attempts    = MAX_ATTEMPTS,
        )
    )


if __name__ == "__main__":
    main()








article = "Macroeconomic headwinds in 2022 caused a stock market sell-off that affected countless industries. The Nasdaq Compositec " \
"index tumbled 33% throughout the year. Tech stocks were some of the hardest hit, as reductions in consumer spending meant" \
"multiple quarters of dismal earnings. However, excitement over high-growth industries like artificial intelligence (AI) has" \
"triggered a recovery in 2023 and illustrated why a market downturn could be the best time to make a long-term investment in" \
"tech stocks. The Nasdaq Composite has surged 41% year to date, rewarding those who either held or bought at the bottom. As" \
"a result, it's not a bad idea to get familiar with some of the best companies to invest in during a sell-off and be prepared " \
"to strike when the time is right. Here are three stocks you can confidently buy after a market downturn. 1. Nvidia According" \
"to research firm Gartner, PC shipments fell 16% year over year in 2022. Spikes in inflation led to reductions in consumer" \
"spending on tech, with chipmakers hit hard. As a result, shares in Nvidia (NASDAQ: NVDA) plunged 50 percent in the 12 months leading" \
"to 2023. However, the company came back better than ever this year, with its stock up 231 percent since Jan. 1 as its earnings hit" \
"new heights. The company profited from a boom in AI, which sent chip demand soaring. Nvidia's graphics processing units (GPUs)" \
"have become the preferred hardware for AI developers everywhere, with the company's revenue climbing 206% year over year in its" \
"most recent quarter (third quarter of fiscal 2024). Data by YCharts As a leading chipmaker, Nvidia supplies its GPUs to markets" \
"across tech. The company is an excellent option in a market downturn, as its stock will likely soar over the long term." \
"Additionally, the chart shows Nvidia's price-to-earnings ratio (P/E) and price-to-free cash flow ratio have both plunged since" \
"July, meaning its shares are currently trading at their cheapest position in months. 2. Apple As the world's most valuable company," \
"with a market cap above $3 trillion, Apple's (NASDAQ: AAPL) stock rarely goes on sale. In fact, the table shows the iPhone maker" \
"outperformed many of the biggest names in tech throughout 2022. Apple's performance amid economic challenges proved its resilience," \
"as it became a haven for many investors. Data by YCharts In 2023, Apple has once again showcased its consistency. Macro headwinds" \
"have caught up with the company, as pullback from consumers led to repeated declines in product sales and a 3% year-over-year dip" \
"in revenue for fiscal 2023. Yet loyal investors continued to believe in its long-term growth, with Apple's stock up 52% year to date." \
"The company's nearly $100 billion in free cash flow, popular range of products and services, and considerable brand loyalty from" \
"consumers make it challenging to question Apple's ability to flourish over the next five to 10 years. Apple remains the biggest" \
"name in consumer tech and the home of a digital services business that posted revenue growth of 9% this year. Services are another" \
"reason you can confidently invest in the tech giant, with the App Store and platforms like Apple TV+ hitting profit margins of more" \
"than 70%. Moreover, Apple's stock has risen 377% over the last five years. Even if the company delivers half that growth over the next" \
"five years, it will still more than double the stock growth of competitors Amazon or Alphabet since 2018. As a result, a market downturn" \
"could be the perfect time to invest in this tech company and buy its stock at a bargain. 3. Microsoft Like Apple, Microsoft's (NASDAQ: MSFT)" \
"stock often trades at a premium. However, years of consistency and stellar gains make it worth its high price tag, especially in a sell-off." \
"The company has become a tech behemoth, with brands such as Windows, Office, Azure, and Xbox granting it lucrative positions in multiple" \
"industries. Shares in Microsoft gained 55% this year after tumbling amid last year's market downturn. The tech giant has rallied Wall Street" \
"by heavily investing in AI. A close partnership with ChatGPT developer OpenAI allowed Microsoft to introduce AI upgrades across its product" \
"lineup as it seeks to become the go-to for consumers and businesses everywhere seeking ways to integrate AI into their daily workflows." \
"Microsoft has significant potential in AI, with the market projected to develop at a compound annual growth rate of 37% until at least 2030." \
"As a result, leading positions in productivity software and cloud computing could see Microsoft profit significantly from the sector as" \
"it expands its AI offerings. Data by YCharts This chart shows Microsoft's forward P/E and price-to-free cash flow are high at 33 and 44," \
"indicating its stock isn't exactly a bargain. However, both figures are significantly lower than those of other companies active in AI." \
"Microsoft is a company you can confidently invest in at almost any time, but especially during a market downturn. Should you invest $1,000" \
"in Nvidia right now? Before you buy stock in Nvidia, consider this: The Motley Fool Stock Advisor analyst team just identified what" \
"they believe are the 10 best stocks for investors to buy now... and Nvidia wasn't one of them. The 10 stocks that made the cut could" \
"produce monster returns in the coming years. Stock Advisor provides investors with an easy-to-follow blueprint for success, including" \
"guidance on building a portfolio, regular updates from analysts, and two new stock picks each month. The Stock Advisor service has more" \
"than tripled the return of S&P 500 since 2002*. See the 10 stocks *Stock Advisor returns as of December 11, 2023 Suzanne Frey, an" \
"executive at Alphabet, is a member of The Motley Fool’s board of directors. John Mackey, former CEO of Whole Foods Market, an Amazon" \
"subsidiary, is a member of The Motley Fool’s board of directors. Randi Zuckerberg, a former director of market development and spokeswoman" \
"for Facebook and sister to Meta Platforms CEO Mark Zuckerberg, is a member of The Motley Fool's board of directors. Dani Cook has no" \
"position in any of the stocks mentioned. The Motley Fool has positions in and recommends Advanced Micro Devices, Alphabet, Amazon, Apple," \
"Meta Platforms, Microsoft, and Nvidia. The Motley Fool recommends Gartner. The Motley Fool has a disclosure policy. The views and opinions" \
"expressed herein are the views and opinions of the author and do not necessarily reflect those of Nasdaq, Inc."

answer = ['AMD', 'NVDA', 'MSFT', 'AMZN']