import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import re
import ast
from openai import OpenAI
import time
import pandas as pd
import shutil
import asyncio
from tqdm import tqdm
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
    :param provider: One of 'openai' or 'groq'.
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
        "deepseek-r1-distill-llama-70b": 128000,
        "qwen-qwq-32b": 8192*2,
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

    elif provider == "openai":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        model = model or "gpt-4-turbo"

    else:
        raise ValueError("Provider must be either 'openai' or 'groq'")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )

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
            article_text = row['Article']
            related_tickers = extract_relevant_tickers(primary_ticker, article_text)
            return idx, related_tickers


    for idx, row in df.iterrows():
        if pd.isna(row['Associated_tickers']) or row['Associated_tickers'] == "[]":
            tasks.append(process_row(idx, row))

        if len(tasks) >= 100:
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




if __name__ == "__main__":
    if not os.path.exists(config.NEWS_CSV_PATH_ASSOCIATED_TICKERS):
        shutil.copy(config.NEWS_CSV_PATH_CLEAN, config.NEWS_CSV_PATH_ASSOCIATED_TICKERS)

    df = pd.read_csv(config.NEWS_CSV_PATH_ASSOCIATED_TICKERS)

    asyncio.run(
        populate_associated_tickers_column_async(
            df=df,
            output_path=config.NEWS_CSV_PATH_ASSOCIATED_TICKERS,
            concurrency_limit=20  # adjust based on provider limits
        )
    )

