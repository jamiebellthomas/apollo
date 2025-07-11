import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import re
import ast
from openai import OpenAI, AsyncOpenAI
import time
import pandas as pd
import shutil
import asyncio
from tqdm import tqdm
import ollama
from ticker_keyword_lookup import TICKER_KEYWORD_LOOKUP
CURRENT_MODEL_INDEX = 0



def generate_related_ticker_prompt(primary_ticker: str, article_text: str) -> str:
    """
    Generate a prompt for an LLM to extract relevant secondary tickers from a news article.

    Args:
        primary_ticker (str): The main stock ticker associated with the article.
        article_text (str): The full text of the news article.

    Returns:
        str: A formatted prompt string.
    """
    prompt = f"""
    You are a financial NLP assistant. Given the following financial news article, identify all stock tickers that are relevant to the article content.
    Do NOT include the primary ticker: {primary_ticker}
    Only return a valid Python list of ticker symbols (e.g., ['MSFT', 'NVDA', 'TSLA']) - any other format will be deemed wrong and rejected.
    There can be no other companies associated with an article in which case just return an empty single element ([''])
    And remember the tickers NEED quotations indicating they are strings or the formal WILL BE INVALID
    ARTICLE:
    \"\"\"{article_text}\"\"\"
    """
    return prompt.strip()


def extract_ticker_list_from_response(response: str) -> tuple[list,bool]:
    """
    Extract a Python list of tickers from an LLM response string.

    Args:
        response (str): The full string returned by the LLM.

    Returns:
        list[str] or None: A list of tickers if found, otherwise None.
    """
    match = re.search(r"\[.*?\]", response)
    if match:
        try:
            return (ast.literal_eval(match.group(0)),True)
        except Exception as e:
            print(f"[ERROR] Failed to parse list: {e}")
            return (response,False)
    return (response,False)



def extract_relevant_tickers_openai(prompt: str, model: str, provider: str = "groq") -> tuple[list,bool]:
    """
    Extract basic and diluted EPS from a structured prompt using either OpenAI or Groq.
    
    :param prompt: The prompt to send to the model.
    :param provider: One of 'openai' or 'groq' or ollama.
    :param model: Optional model override. Defaults:
                  - 'gpt-4-turbo' for OpenAI
                  - 'llama3-70b-8192' for Groq
    :return: A list of extracted EPS strings.
    """

    model_token_limits = {
        "llama3-70b-8192": 8192,
        "llama3-8b-8192": 8192,
        "mixtral-8x7b-32768": 32768,
        "gemma-7b-it": 8192,
        "llama-3.3-70b-versatile": 32768,
        "deepseek-r1-distill-llama-70b": 131072,
        "qwen-qwq-32b": 8192*2,
        "mistral-saba-24b":32768
    }

    

    if provider == "groq":
        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        model = model or "llama3-70b-8192"
        max_tokens = model_token_limits.get(model,8192)
        original_prompt_length = len(prompt)
        prompt = prompt[:max_tokens]
        new_prompt_length = len(prompt)
        # Print percentage reduction in prompt length if reduced
        if original_prompt_length > new_prompt_length:
            reduction_percentage = ((original_prompt_length - new_prompt_length) / original_prompt_length) * 100
            print(f"[INFO] Original prompt length: {original_prompt_length}, Truncated prompt length: {new_prompt_length}, Max tokens for model: {max_tokens}")
            print(f"Reduced by {reduction_percentage:.2f}%")
            if reduction_percentage > 70:
                print("[WARN] Over 80 percent reduction in prompt length, terminating early to avoid potential issues with model understanding.")
                return ""
        
        response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )

    elif provider == "openai":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        model = model or "gpt-4-turbo"
        response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    
    elif provider == "ollama":
        response = ollama.chat(
            model='llama3:8b',
            messages=[
                {'role': 'user', 'content': 'Write me a short poem about the sea.'}
            ]
        )

    else:
        raise ValueError("Provider must be either 'openai' or 'groq'")


    usage = response.usage
    print(f"Prompt tokens: {usage.prompt_tokens}, Completion tokens: {usage.completion_tokens}, Total: {usage.total_tokens}")

    raw_output = response.choices[0].message.content.strip()

    return extract_ticker_list_from_response(response=raw_output)


def extract_relevant_tickers(primary_ticker:str, article_text: str) -> list:
    """
    Extract EPS data from text blocks using model fallback if rate limits are hit.
    """
    global CURRENT_MODEL_INDEX
    models = ["llama3-70b-8192","llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b","qwen-qwq-32b"]

    prompt = generate_related_ticker_prompt(primary_ticker=primary_ticker,
                                            article_text=article_text)

    for attempt in range(5):
        model = models[CURRENT_MODEL_INDEX]
        print(f"[INFO] Attempt {attempt + 1}: Using model {model}")
        try:
            response,success = extract_relevant_tickers_openai(prompt, model=model)
            print(f"[INFO] Received response: {response}")

            if success:
                return response

            print("[WARN] Invalid EPS format, retrying...")

        except Exception as e:
            if '429' in str(e):
                print(str(e))
                if CURRENT_MODEL_INDEX == len(models) - 1:
                    print(f"[WARN] Rate limit hit for all models, waiting 60 seconds before retrying last model: {models[CURRENT_MODEL_INDEX]}")
                    time.sleep(60)  # Wait for a minute before retrying the last model to let TPM reset
                else:
                    CURRENT_MODEL_INDEX = (CURRENT_MODEL_INDEX + 1) % len(models)
                
                    print(f"[WARN] Rate limit hit, switching model to {models[CURRENT_MODEL_INDEX]}")
                
            else:
                print(f"[ERROR] Unexpected error: {e}")
                break

            # Add something for timeout error (like wait 2 mins so we dont skip rows in for loop)

    return []





async def populate_associated_tickers_column_async(df: pd.DataFrame, output_path: str, concurrency_limit: int = 5) -> pd.DataFrame:
    if 'Associated_tickers' not in df.columns:
        df['Associated_tickers'] = None

    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = []
    updated_indices = set()

    total_rows = len(df)
    overall_pbar = tqdm(total=total_rows, desc="Overall progress", position=0)

    async def process_row(idx, row):
        async with semaphore:
            primary_ticker = row['Stock_symbol']
            article_text   = row['Article']

            # OLD (blocks everything)
            # related_tickers = extract_relevant_tickers(primary_ticker, article_text)

            # NEW (runs in background thread; this coroutine can now move on)
            related_tickers = await asyncio.to_thread(
                extract_relevant_tickers, primary_ticker, article_text
            )
            return idx, related_tickers


    for idx, row in df.iterrows():
        if pd.isna(row['Associated_tickers']) or row['Associated_tickers'] == "[]":
            tasks.append(process_row(idx, row))

        if len(tasks) >= 50:
            results = await asyncio.gather(*tasks)
            for idx_, related_tickers in results:
                df.at[idx_, 'Associated_tickers'] = related_tickers
                updated_indices.add(idx_)
                overall_pbar.update(1)

            print(f"[INFO] Saving progress → {output_path}")
            df.to_csv(output_path, index=False)
            tasks.clear()
        
        else:
            overall_pbar.update(1)


    # Process any remaining tasks
    if tasks:
        batch_pbar = tqdm(total=len(tasks), desc="Final batch", position=1)
        results = await asyncio.gather(*tasks)
        for idx_, related_tickers in results:
            df.at[idx_, 'Associated_tickers'] = related_tickers
            updated_indices.add(idx_)
            batch_pbar.update(1)
            overall_pbar.update(1)

        print(f"[INFO] Final save → {output_path}")
        df.to_csv(output_path, index=False)

    overall_pbar.close()
    return df






# ───── helpers ───────────────────────────────────────────────────────
SPACES = re.compile(r"\s+")

def _token_pattern(term: str) -> str:
    """
    Turn a lookup term into a regex fragment.
    • 1-letter ticker (“V”)  → require non-dot boundary on the left.
    • Otherwise             → exact case, allow flexible whitespace.
    """
    if len(term) == 1:                              # e.g. "V"
        return rf"(?<![\w\.]){term}(?![\w])"
    return re.escape(term).replace(r"\ ", r"\s+")

def _build_regex(term_to_tickers: dict[str, set[str]]) -> re.Pattern:
    """Compile one big case-sensitive regex for all terms."""
    frags = [_token_pattern(t) for t in term_to_tickers]
    frags.sort(key=len, reverse=True)               # longest first
    return re.compile("|".join(frags))              # ← NO IGNORECASE
# ────────────────────────────────────────────────────────────────────


def populate_associated_tickers_with_regex(
        df: pd.DataFrame,
        text_col: str = "Article",
        symbol_col: str = "Stock_symbol",
        target_col: str = "Associated_tickers",
):
    """Add a list of secondary tickers to `target_col` (strict case)."""

    # 1️ build term → ticker(s)  (keep ALL tickers, even IT/ON)
    term_to_tickers: dict[str, set[str]] = {}
    for ticker, terms in TICKER_KEYWORD_LOOKUP.items():
        for term in terms:
            key = SPACES.sub(" ", term)             # collapse weird spaces
            term_to_tickers.setdefault(key, set()).add(ticker)

    # 2️ compile regex once
    pattern = _build_regex(term_to_tickers)

    # 3️ ensure output column exists
    if target_col not in df.columns:
        df[target_col] = None

    # 4️ iterate rows with progress + autosave every 20 000
    total_rows = len(df)
    for idx, row in df.iterrows():
        primary = str(row[symbol_col]).upper()
        text    = str(row[text_col])

        matches = {
            ticker
            for m in pattern.finditer(text)
            for ticker in term_to_tickers[SPACES.sub(" ", m.group(0))]
            if ticker != primary                       # exclude the row’s own
        }

        df.at[idx, target_col] = list(matches) if matches else "[]"

        # ---- milestone every 20 000 rows ----
        if (idx + 1) % 20_000 == 0 or idx == total_rows - 1:
            pct = 100 * (idx + 1) / total_rows
            print(f"[{idx + 1:,}/{total_rows:,}]  {pct:5.1f}% complete — autosaving …")
            df.to_csv(config.NEWS_CSV_PATH_ASSOCIATED_TICKERS, index=False)

    return df

def clear_duplicate_articles(df:pd.DataFrame):

    print(f"Original length: {len(df)}")

    # Drop any columns whose *name* contains 'Unnamed'
    df = df.drop(columns=[col for col in df.columns if "Unnamed" in col])


    # Drop duplicate Article_title rows, keeping the first occurrence
    df_unique = df.drop_duplicates(subset="Article_title", keep="first")

    print(f"New length: {len(df_unique)}")

    df_unique.to_csv(config.NEWS_CSV_PATH_ASSOCIATED_TICKERS, index=False)

    





if __name__ == "__main__":
    # if not os.path.exists(config.NEWS_CSV_PATH_ASSOCIATED_TICKERS):
    #     shutil.copy(config.NEWS_CSV_PATH_CLEAN, config.NEWS_CSV_PATH_ASSOCIATED_TICKERS)

    # df = pd.read_csv(config.NEWS_CSV_PATH_ASSOCIATED_TICKERS)

    # asyncio.run(
    #     populate_associated_tickers_column_async(
    #         df=df,
    #         output_path=config.NEWS_CSV_PATH_ASSOCIATED_TICKERS,
    #         concurrency_limit=100  # adjust based on provider limits
    #     )
    # )
    # populate_associated_tickers_with_regex(df=df)
    related_ticker_db = pd.read_csv(config.NEWS_CSV_PATH_ASSOCIATED_TICKERS)
    clear_duplicate_articles(df=related_ticker_db)



"""
You are a financial NLP assistant. Given the following financial news article, identify all stock tickers that are relevant to the article content.
Do NOT include the primary ticker: AAPL
Only return a valid Python list of ticker symbols (e.g., ['MSFT', 'NVDA', 'TSLA']) - any other format will be deemed wrong and rejected.
There can be no other companies associated with an article in which case just return an empty single element ([''])
And remember the tickers NEED quotations indicating they are strings or the formal WILL BE INVALID
ARTICLE:
"Macroeconomic headwinds in 2022 caused a stock market sell-off that affected countless industries. The Nasdaq Composite 
index tumbled 33% throughout the year. Tech stocks were some of the hardest hit, as reductions in consumer spending meant 
multiple quarters of dismal earnings. However, excitement over high-growth industries like artificial intelligence (AI) has 
triggered a recovery in 2023 and illustrated why a market downturn could be the best time to make a long-term investment in 
tech stocks. The Nasdaq Composite has surged 41% year to date, rewarding those who either held or bought at the bottom. As 
a result, it's not a bad idea to get familiar with some of the best companies to invest in during a sell-off and be prepared 
to strike when the time is right. Here are three stocks you can confidently buy after a market downturn. 1. Nvidia According 
to research firm Gartner, PC shipments fell 16% year over year in 2022. Spikes in inflation led to reductions in consumer 
spending on tech, with chipmakers hit hard. As a result, shares in Nvidia (NASDAQ: NVDA) plunged 50% in the 12 months leading 
to 2023. However, the company came back better than ever this year, with its stock up 231% since Jan. 1 as its earnings hit 
new heights. The company profited from a boom in AI, which sent chip demand soaring. Nvidia's graphics processing units (GPUs) 
have become the preferred hardware for AI developers everywhere, with the company's revenue climbing 206% year over year in its 
most recent quarter (third quarter of fiscal 2024). Data by YCharts As a leading chipmaker, Nvidia supplies its GPUs to markets 
across tech. The company is an excellent option in a market downturn, as its stock will likely soar over the long term. 
Additionally, the chart shows Nvidia's price-to-earnings ratio (P/E) and price-to-free cash flow ratio have both plunged since 
July, meaning its shares are currently trading at their cheapest position in months. 2. Apple As the world's most valuable company, 
with a market cap above $3 trillion, Apple's (NASDAQ: AAPL) stock rarely goes on sale. In fact, the table shows the iPhone maker 
outperformed many of the biggest names in tech throughout 2022. Apple's performance amid economic challenges proved its resilience, 
as it became a haven for many investors. Data by YCharts In 2023, Apple has once again showcased its consistency. Macro headwinds 
have caught up with the company, as pullback from consumers led to repeated declines in product sales and a 3% year-over-year dip 
in revenue for fiscal 2023. Yet loyal investors continued to believe in its long-term growth, with Apple's stock up 52% year to date. 
The company's nearly $100 billion in free cash flow, popular range of products and services, and considerable brand loyalty from 
consumers make it challenging to question Apple's ability to flourish over the next five to 10 years. Apple remains the biggest 
name in consumer tech and the home of a digital services business that posted revenue growth of 9% this year. Services are another 
reason you can confidently invest in the tech giant, with the App Store and platforms like Apple TV+ hitting profit margins of more 
than 70%. Moreover, Apple's stock has risen 377% over the last five years. Even if the company delivers half that growth over the next 
five years, it will still more than double the stock growth of competitors Amazon or Alphabet since 2018. As a result, a market downturn 
could be the perfect time to invest in this tech company and buy its stock at a bargain. 3. Microsoft Like Apple, Microsoft's (NASDAQ: MSFT) 
stock often trades at a premium. However, years of consistency and stellar gains make it worth its high price tag, especially in a sell-off. 
The company has become a tech behemoth, with brands such as Windows, Office, Azure, and Xbox granting it lucrative positions in multiple 
industries. Shares in Microsoft gained 55% this year after tumbling amid last year's market downturn. The tech giant has rallied Wall Street 
by heavily investing in AI. A close partnership with ChatGPT developer OpenAI allowed Microsoft to introduce AI upgrades across its product 
lineup as it seeks to become the go-to for consumers and businesses everywhere seeking ways to integrate AI into their daily workflows. 
Microsoft has significant potential in AI, with the market projected to develop at a compound annual growth rate of 37% until at least 2030. 
As a result, leading positions in productivity software and cloud computing could see Microsoft profit significantly from the sector as 
it expands its AI offerings. Data by YCharts This chart shows Microsoft's forward P/E and price-to-free cash flow are high at 33 and 44, 
indicating its stock isn't exactly a bargain. However, both figures are significantly lower than those of other companies active in AI. 
Microsoft is a company you can confidently invest in at almost any time, but especially during a market downturn. Should you invest $1,000 
in Nvidia right now? Before you buy stock in Nvidia, consider this: The Motley Fool Stock Advisor analyst team just identified what 
they believe are the 10 best stocks for investors to buy now... and Nvidia wasn't one of them. The 10 stocks that made the cut could 
produce monster returns in the coming years. Stock Advisor provides investors with an easy-to-follow blueprint for success, including 
guidance on building a portfolio, regular updates from analysts, and two new stock picks each month. The Stock Advisor service has more 
than tripled the return of S&P 500 since 2002*. See the 10 stocks *Stock Advisor returns as of December 11, 2023 Suzanne Frey, an 
executive at Alphabet, is a member of The Motley Fool’s board of directors. John Mackey, former CEO of Whole Foods Market, an Amazon 
subsidiary, is a member of The Motley Fool’s board of directors. Randi Zuckerberg, a former director of market development and spokeswoman 
for Facebook and sister to Meta Platforms CEO Mark Zuckerberg, is a member of The Motley Fool's board of directors. Dani Cook has no 
position in any of the stocks mentioned. The Motley Fool has positions in and recommends Advanced Micro Devices, Alphabet, Amazon, Apple, 
Meta Platforms, Microsoft, and Nvidia. The Motley Fool recommends Gartner. The Motley Fool has a disclosure policy. The views and opinions 
expressed herein are the views and opinions of the author and do not necessarily reflect those of Nasdaq, Inc."
"""
# ANSWER:
"['NVDA', 'MSFT']"