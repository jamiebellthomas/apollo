# Right now, this script will be used to experiment with the edgartools package
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from sec_downloader import Downloader
import sec_parser as sp
import openai
import pandas as pd
import re


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

        if "earnings per share" in node.text.lower() or "earnings per common share" in node.text.lower() or "earnings (loss) per common share" in node.text.lower():
            # remove and mentions of "\xa0" from the text
        
            temp_text = node.text.replace("\xa0", " ")
            text.append(temp_text)

    return text

def extract_eps_data(text: list) -> str:
    """
    This takes the list of text strings and extracts the earnings per share (EPS) data using 
    an OpenAI model.
    It returns a JSON string with the basic and diluted EPS values.
    """


    # Make a query to OpenAI to summarize the earnings per share information, getting both basic and diluted EPS
    client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    prompt = "I am about to send you all components relating to the earnings per share of a company. " \
            "Please summarize the earnings per share information, including both basic and diluted EPS. " \
            "I want you to return it as two separate values. The first value should be the basic EPS, " \
            "and the second value should be the diluted EPS. If either is not available, return None for that " \
            "value. This must be returned in a specific format -> 'basic_eps: value, diluted_eps: value'.  " \
            "It is very important you stick to this formatting otherwise the output is invalid" \
            " the information: " + " ".join(text)


    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt}
        ],
            temperature=0.6
        )

        # Extract and split into list
    raw_output = response.choices[0].message.content.strip()

    valid_format = check_eps_formatting(raw_output)
    count = 0

    while not valid_format:
        count += 1
        if count > 5:
            raise ValueError("Exceeded maximum retries for valid EPS format.")
        # If the format is not valid, retry with a new prompt
        print("Invalid format detected. Retrying with a new prompt.")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        raw_output = response.choices[0].message.content.strip()
        valid_format = check_eps_formatting(raw_output) 

    return raw_output

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
        eps_json = extract_eps_data(eps_text_data)
        data = extract_eps_from_openai_result(eps_json)
        basic_eps, diluted_eps = data
        return (basic_eps, diluted_eps)
    else:
        return (0.0, 0.0)
    



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
    
def main():
    """
    Main function to run the EPS extraction process.
    """
    # 1. Open config.FILING_DATES_AND_URLS_CSV
    original_df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)

    # 2. Create new DataFrame with only the necessary columns (Ticker, Form Type, URL) and save
    df = original_df[['Ticker', 'Form Type', 'URL']].copy()
    df.to_csv(config.EPS_DATA_CSV, index=False)

    # 3. If there is no 'Accession Number' column, create it
    # This will be taken from the URLs which are in the format:
    # https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm
    # And convert it to the format: 
    # 0000320193-24-000081
    if 'Accession Number' not in df.columns:
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

    # Save the DataFrame to the CSV
    df.to_csv(config.EPS_DATA_CSV, index=False)
    
    # Now we will iterate over each row and apply the extract_eps function to the 'Query' column
    # We will also keep track of the rows that failed to process, so we can retry them later
    print("[INFO] Starting EPS extraction process...")
    # Initialize a list to keep track of bad row indices
    total_rows = len(df)
    bad_row_indices = []
    for index, row in df.iterrows():
        # If all eps columns are already empty, do the try loop

        if index % 10 == 0:  # Print every 100 rows
            percent = (index + 1) / total_rows * 100
            print(f"[INFO] Processing row {index + 1}/{total_rows} ({percent:.2f}%), Saving progress...")
            df.to_csv(config.EPS_DATA_CSV, index=False)  # Save progress

        # If the row already has any eps values, skip it
        if (row['quarterly_raw_eps'] is not None or row['quarterly_diluted_eps'] is not None or
            row['annual_raw_eps'] is not None or row['annual_diluted_eps'] is not None):
            continue
       
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

    print(f"[INFO] Finished processing {total_rows} rows.")
    # Save the updated DataFrame to the CSV
    df.to_csv(config.EPS_DATA_CSV, index=False)
    if len(bad_row_indices) > 0:
        print(f"[WARNING] Some rows failed to process: {bad_row_indices}")
    else:
        print("[INFO] All rows processed successfully.")

if __name__ == "__main__":
    main()




    


    

    


