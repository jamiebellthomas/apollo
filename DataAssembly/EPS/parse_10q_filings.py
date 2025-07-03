import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

from sec_downloader import Downloader
import sec_parser as sp
import pandas as pd
import re
import warnings
import time
from openai import OpenAI


warnings.filterwarnings("ignore", category=UserWarning)

CURRENT_MODEL_INDEX = 0

def check_eps_formatting(eps_data: str) -> bool:
    """
    Check if the EPS data is in the correct format.
    The expected format is 'basic_eps: value, diluted_eps: value'.
    Accepts negative numbers and 'None' as valid values.
    """
    if not eps_data:
        return False

    parts = eps_data.split(',')
    if len(parts) != 2:
        return False

    basic_eps_part = parts[0].strip()
    diluted_eps_part = parts[1].strip()

    if not basic_eps_part.startswith("basic_eps: ") or not diluted_eps_part.startswith("diluted_eps: "):
        return False

    basic_eps_value = basic_eps_part.split(":", 1)[1].strip()
    diluted_eps_value = diluted_eps_part.split(":", 1)[1].strip()

    def is_valid_number(val: str) -> bool:
        if val.lower() == "none":
            return True
        try:
            float(val)
            return True
        except ValueError:
            return False

    return is_valid_number(basic_eps_value) and is_valid_number(diluted_eps_value)


def extract_eps_from_openai_result(eps_data: str) -> tuple[float, float]:
    """
    Extracts the basic and diluted EPS values from the OpenAI result string.
    Returns a tuple of (basic_eps, diluted_eps).
    If a value is None, it will be returned as None.
    """
    parts = eps_data.split(',')
    basic_eps = parts[0].split(':')[1].strip()
    diluted_eps = parts[1].split(':')[1].strip()

    # Convert to float or None
    basic_eps = float(basic_eps) if basic_eps.lower() != "none" else None
    diluted_eps = float(diluted_eps) if diluted_eps.lower() != "none" else None

    return basic_eps, diluted_eps


def build_eps_prompt(text: list[str]) -> str:
    return (
        "I am about to send you all components relating to the earnings per share of a company. "
        "Please summarize the earnings per share information, including both basic and diluted EPS. "
        "I want you to return it as two separate values. The first value should be the basic EPS, "
        "and the second value should be the diluted EPS. If either is not available, return None for that "
        "value. This must be returned in a specific format -> 'basic_eps: value, diluted_eps: value'. "
        "It is very important you stick to this formatting otherwise the output is invalid. "
        "DO NOT return any other text, just the values in the format specified. "
        "For quarterly filings, the EPS is for the most recent quarter - NOT the rest of the year (e.g Six Months Ended / Nine Months Ended). "
        "It is also very important you spot negative EPS values, and return them as negative numbers. These are usually denoted by loss in the tables and may potentially be in brackets. "
        "The information: "
        + " ".join(text)
        + " AGAIN, DO NOT return any other text, just the values in the format specified and only the EPS data from the most recent quarter."
    )

            
    

def extract_relevant_eps_data_html(query: str) -> str:
    """
    Query is a string in the format 'Ticker/AccessionNumber'
    This function returns the HTML content of the filing.
    """

    dl = Downloader(config.COMPANY_NAME, config.EMAIL)
    html = dl.get_filing_html(query=query)


    elements: list = sp.Edgar10QParser().parse(html)

    tree = sp.TreeBuilder().build(elements)
    text = []
    for node in tree.nodes:
        

        lowered_text = node.text.lower()

        if (re.search(r"earnings\s+.*?\s+per\s+share", lowered_text) or 
            re.search(r"earnings\s+.*?\s+per\s+common\s+share", lowered_text) or
            re.search(r"income\s+.*?\s+per\s+share", lowered_text) or
            "earnings per share" in node.text.lower() or 
            "earnings per common share" in node.text.lower() or
            "income per share" in node.text.lower() or 
            "income per common share" in node.text.lower() or
            "net income (loss) per share of common stock" in node.text.lower() or
            "net income per share of common stock" in node.text.lower()
            ):

            node_type = str(node._semantic_element)
            # If the node type contains 'TextElement' skip it
            if 'TextElement' in node_type or 'TopSectionTitleElement' in node_type:
                continue

            # If it's a TitleElement, get the child nodes
            elif 'TitleElement' in node_type:

                temp_text = node.text.replace("\xa0", " ")
                # replace \u200b with a space
                temp_text = temp_text.replace("\u200b", " ")
                # replace \n with nothing
                temp_text = temp_text.replace("\n", "")
                text.append(temp_text)

                for child in node.get_descendants():
                    # print(f"[DEBUG] Found child node: {child._semantic_element}")
                    if ('TableElement' in str(child._semantic_element)):
                        
                        # print(f"\t[DEBUG] Found TableElement child: {child._semantic_element}")
                        # remove and mentions of "\xa0" from the text
                        temp_text = child.text.replace("\xa0", " ")
                        # replace \u200b with a space
                        temp_text = temp_text.replace("\u200b", " ")
                        # replace \n with nothing
                        temp_text = temp_text.replace("\n", "")
                        text.append(temp_text)
                        # print(f"\t[DEBUG] Extracted node typeee: {child._semantic_element}, text: {temp_text}")
                    else: 
                        continue    
                
                
            
            else:
                temp_text = node.text.replace("\xa0", " ")
                # replace \u200b with a space
                temp_text = temp_text.replace("\u200b", " ")
                # replace \n with nothing
                temp_text = temp_text.replace("\n", "")
                text.append(temp_text)

    # Remove any empty strings from the list
    text = [t for t in text if t.strip()]
    print(f"[INFO] Extracted {len(text)} text blocks, using all for prompt.")
    return text

def extract_eps_from_llm_output(raw_output: str) -> str:
    """
    Extract the correct EPS-formatted output from the end of a noisy LLM response.

    Returns only the part matching: 'basic_eps: <float>, diluted_eps: <float>',
    where either float can be negative, and may be of any digit length.

    If no valid pattern is found, returns the raw_output unchanged.
    """
    # Remove any surrounding whitespace and normalize brackets used for negatives
    raw_output = raw_output.strip().replace('(', '-').replace(')', '')

    # Regex to match 'basic_eps: <float>, diluted_eps: <float>' at the end of the string
    pattern = r"(basic_eps:\s*-?\d+(?:\.\d+)?\s*,\s*diluted_eps:\s*-?\d+(?:\.\d+)?)\s*$"

    match = re.search(pattern, raw_output, re.IGNORECASE)
    if match:
        return match.group(1)

    return raw_output  # let the formatting checker handle rejections if needed


def extract_eps_openai(prompt: str, model: str, provider: str = "groq") -> str:
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
    # remove and brackets from the output
    raw_output = re.sub(r'[\[\]()]', '', raw_output)
    # remove dollar signs
    raw_output = re.sub(r'\$', '', raw_output)

    return extract_eps_from_llm_output(raw_output=raw_output)


def extract_eps_data(text_blocks: list[str]) -> tuple[float | None, float | None]:
    """
    Extract EPS data from text blocks using model fallback if rate limits are hit.
    """
    global CURRENT_MODEL_INDEX
    models = ["llama3-70b-8192","llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b","qwen-qwq-32b"]
    # ------------ Apply for research credits on OpenAI --------------
    # Work out how many CPUs, how much RAM and disc space I will need

    full_text = "\n\n".join(text_blocks)
    prompt = build_eps_prompt(full_text)

    for attempt in range(5):
        model = models[CURRENT_MODEL_INDEX]
        print(f"[INFO] Attempt {attempt + 1}: Using model {model}")
        try:
            response = extract_eps_openai(prompt, model=model).strip()
            print(f"[INFO] Received response: {response}")

            if check_eps_formatting(response):
                return extract_eps_from_openai_result(response)

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

    return (None, None)

def extract_eps(query:str) -> tuple[float, float]:

    """
    Main function to extract EPS data from a 10-Q or 10-K filing.
    :param query: A string in the format 'Ticker/AccessionNumber'
    :return: A string containing the basic and diluted EPS values in the format 'basic_eps: value, diluted_eps: value'.
    """

    # Sample 10-Q Filing
    # query = "AAPL/0000320193-24-000069"
    # Sample 10-K Filing
    #query = "AAPL/0000320193-24-000123"

    # Extract relevant EPS data from the HTML content
    eps_text_data = extract_relevant_eps_data_html(query=query)

    if eps_text_data:
        # Extract EPS values using OpenAI
        data = extract_eps_data(eps_text_data)
        basic_eps, diluted_eps = data
        return (basic_eps, diluted_eps)
    else:
        return (None, None)
    



def convert_sec_url_to_accession(url: str) -> str:
    """
    Converts a SEC EDGAR filing URL into the accession number format ##########-##-######.

    Example:
    Input:  https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm
    Output: 0000320193-24-000081
    """
    match = re.search(r'/data/(\d+)/(\d{10})(\d{2})(\d{6})/', url)
    if not match:
        raise ValueError("URL does not match expected EDGAR format")

    cik, prefix, year, seq = match.groups()
    accession_number = f"{prefix}-{year}-{seq}"
    return accession_number
    
def main(start:int):
    """
    Main function to run the EPS extraction process.
    """
    # 1. Open config.FILING_DATES_AND_URLS_CSV
    original_df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)

    # 2. Create new DataFrame with only the necessary columns (Ticker, Form Type, URL) and save it to config.EPS_DATA_CSV if it doesn't exist
    if not os.path.exists(config.EPS_DATA_CSV):
        print("[INFO] EPS data CSV does not exist. Creating a new one.")
        df = original_df[['Ticker', 'Form Type', 'URL']].copy()
        df.to_csv(config.EPS_DATA_CSV, index=False)
    else:
        df = pd.read_csv(config.EPS_DATA_CSV)
    

    # 3. If there is no 'Accession Number' column, create it
    # This will be taken from the URLs which are in the format:
    # https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm
    # And convert it to the format: 
    # 0000320193-24-000081

    df['Accession Number'] = df['URL'].apply(convert_sec_url_to_accession)
    # Now for each row we will build the query in the format 'Ticker/AccessionNumber'
    df['Query'] = df.apply(lambda row: f"{row['Ticker']}/{row['Accession Number']}", axis=1)
    # Now for each row, apply the extract_eps function to the 'Query' column, if the 'Form Type' column contains '10-Q', save it to the
    # quarterly_raw_eps, and quarterly_diluted_eps column, if it contains '10-K', save it to the annual_raw_eps, and annual_diluted_eps column
    # If there are no columns called quarterly_raw_eps, quarterly_diluted_eps, annual_raw_eps, and annual_diluted_eps, create them
    if 'quarterly_raw_eps' not in df.columns:
        df['quarterly_raw_eps'] = None
    if 'quarterly_diluted_eps' not in df.columns:
        df['quarterly_diluted_eps'] = None
    if 'annual_raw_eps' not in df.columns:
        df['annual_raw_eps'] = None
    if 'annual_diluted_eps' not in df.columns:
        df['annual_diluted_eps'] = None

    
    # Now we will iterate over each row and apply the extract_eps function to the 'Query' column
    # We will also keep track of the rows that failed to process, so we can retry them later
    print("[INFO] Starting EPS extraction process...")
    # Initialize a list to keep track of bad row indices
    total_rows = len(df)
    bad_row_indices = []
    for index, row in df.iterrows():

        # If the row already has any eps values, skip it
        if (
        ('10-Q' in row['Form Type'] and pd.notnull(row['quarterly_raw_eps']) and pd.notnull(row['quarterly_diluted_eps'])) or
        ('10-K' in row['Form Type'] and pd.notnull(row['annual_raw_eps']) and pd.notnull(row['annual_diluted_eps']))
    ):
            continue

        if index < start:
            continue

        print(f"[INFO] Processing row {index + 1}/{total_rows}: {row['Query']}")

        

        if index % 5 == 0:  # Print every 100 rows
            percent = (index + 1) / total_rows * 100
            print(f"[INFO] Processing row {index + 1}/{total_rows} ({percent:.2f}%), Saving progress...")
            df.to_csv(config.EPS_DATA_CSV, index=False)  # Save progress

        
        try:
            query = row['Query']
            # basic_eps, diluted_eps = extract_eps(query)
            if '10-Q' in row['Form Type']:
                basic_eps, diluted_eps = extract_eps(query)
                df.at[index, 'quarterly_raw_eps'] = basic_eps
                df.at[index, 'quarterly_diluted_eps'] = diluted_eps
            elif '10-K' in row['Form Type']:
                #print(f"[INFO] Skipping 10-K filing (for now)")
                continue
                df.at[index, 'annual_raw_eps'] = basic_eps
                df.at[index, 'annual_diluted_eps'] = diluted_eps
        except Exception as e:
            print(f"[ERROR] Failed to process row {index + 1}: {e}")
            bad_row_indices.append((index,query))
            df.at[index, 'quarterly_raw_eps'] = None
            df.at[index, 'quarterly_diluted_eps'] = None
            df.at[index, 'annual_raw_eps'] = None
            df.at[index, 'annual_diluted_eps'] = None
            continue
        print("---------------------")

    print(f"[INFO] Finished processing {total_rows} rows.")
    # Save the updated DataFrame to the CSV
    df.to_csv(config.EPS_DATA_CSV, index=False)
    if len(bad_row_indices) > 0:
        print(f"[WARNING] Some rows failed to process: {bad_row_indices}")
    else:
        print("[INFO] All rows processed successfully.")

if __name__ == "__main__":
    main(start=4160)
    # data = (extract_relevant_eps_data_html("https://www.sec.gov/Archives/edgar/data/820313/000155837024013696/aph-20240930x10q.htm"))
    # for i in data:
    #     print("-------------------")
    #     print(i)
    # print(extract_eps("https://www.sec.gov/Archives/edgar/data/820313/000155837024013696/aph-20240930x10q.htm"))





    


    

    


