# Right now, this script will be used to experiment with the edgartools package
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from sec_downloader import Downloader
import sec_parser as sp
import openai
import pandas as pd
import re
import warnings
import subprocess
import random
from openai import OpenAI

warnings.filterwarnings("ignore", category=UserWarning)


def check_eps_formatting(eps_data: str) -> bool:
    """
    Check if the EPS data is in the correct format.
    The expected format is 'basic_eps: value, diluted_eps: value'.
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

    # Check if the values are valid numbers or None
    basic_eps_value = basic_eps_part.split(":")[1].strip()
    diluted_eps_value = diluted_eps_part.split(":")[1].strip()
    if basic_eps_value.lower() != "none" and not basic_eps_value.replace('.', '', 1).isdigit():
        return False
    if diluted_eps_value.lower() != "none" and not diluted_eps_value.replace('.', '', 1).isdigit():
        return False
    # If all checks passed, the format is correct
    return True

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


def build_eps_prompt(text: str) -> str:
    return (
        "I am about to send you all components relating to the earnings per share of a company. " \
            "Please summarize the earnings per share information, including both basic and diluted EPS. " \
            "I want you to return it as two separate values. The first value should be the basic EPS, " \
            "and the second value should be the diluted EPS. If either is not available, return None for that " \
            "value. This must be returned in a specific format -> 'basic_eps: value, diluted_eps: value'.  " \
            "It is very important you stick to this formatting otherwise the output is invalid" \
            "DO NOT return any other text, just the values in the format specified. " \
            " For quarterly filings, the EPS is for the most recent quarter - not the rest of the year (e.g Six Months Ended / Nine Months Ended)" \
            " the information: " + " ".join(text)
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
        
        
        # If node contains "earnings per share" in its text, print the node

        if ("earnings per share" in node.text.lower() or "earnings per common share" in node.text.lower()):
            node_type = str(node._semantic_element)
            # If the node type contains 'TextElement' skip it
            if 'TextElement' in node_type:
                continue

            # If it's a TitleElement, get the child nodes
            elif 'TitleElement' in node_type:
                for child in node.get_descendants():
                    if 'EmptyElement' in str(child._semantic_element) or 'TextElement' in str(child._semantic_element):
                        continue
                    # remove and mentions of "\xa0" from the text
                    temp_text = child.text.replace("\xa0", " ")
                    # replace \u200b with a space
                    temp_text = temp_text.replace("\u200b", " ")
                    # replace \n with nothing
                    temp_text = temp_text.replace("\n", "")
                    text.append(temp_text)
            
            else:
                temp_text = node.text.replace("\xa0", " ")
                # replace \u200b with a space
                temp_text = temp_text.replace("\u200b", " ")
                # replace \n with nothing
                temp_text = temp_text.replace("\n", "")
                text.append(temp_text)
    return text


import openai

def extract_eps_openai(prompt: str, provider: str = "groq", model: str = None) -> list[str]:
    """
    Extract basic and diluted EPS from a structured prompt using either OpenAI or Groq.
    
    :param prompt: The prompt to send to the model.
    :param provider: One of 'openai' or 'groq'.
    :param model: Optional model override. Defaults:
                  - 'gpt-4-turbo' for OpenAI
                  - 'llama3-70b-8192' for Groq
    :return: A list of extracted EPS strings.
    """

    if provider == "groq":
        client = OpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        model = model or "llama3-70b-8192"

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

    raw_output = response.choices[0].message.content.strip()
    return raw_output


def extract_eps_data(text_blocks: list[str]) -> str:
    full_text = "\n\n".join(text_blocks)
    prompt = build_eps_prompt(full_text)

    for attempt in range(5):
        response = extract_eps_openai(prompt).strip()
        print(f"[INFO] Attempt {attempt+1}: Received response from OpenAI: {response}")
        if check_eps_formatting(response):
            return extract_eps_from_openai_result(response)
        print(f"[WARN] Invalid format on attempt {attempt+1}, retrying...")
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
    print(f"[INFO] Extracted {len(eps_text_data)} relevant blocks from the filing.")

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
            basic_eps, diluted_eps = extract_eps(query)
            if '10-Q' in row['Form Type']:
                df.at[index, 'quarterly_raw_eps'] = basic_eps
                df.at[index, 'quarterly_diluted_eps'] = diluted_eps
            elif '10-K' in row['Form Type']:
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
    main(start=0)
    # data = (extract_relevant_eps_data_html("https://www.sec.gov/Archives/edgar/data/820313/000155837024013696/aph-20240930x10q.htm"))
    # for i in data:
    #     print("-------------------")
    #     print(i)
    # print(extract_eps("https://www.sec.gov/Archives/edgar/data/820313/000155837024013696/aph-20240930x10q.htm"))





    


    

    


