# Right now, this script will be used to experiment with the edgartools package
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


from edgar import *
import pandas as pd
from edgar.xbrl import XBRLS

from sec_downloader import Downloader
import sec_parser as sp
import openai


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


    html = dl.get_filing_html(query=query)


    elements: list = sp.Edgar10QParser().parse(html)

    tree = sp.TreeBuilder().build(elements)
    text = []
    for node in tree.nodes:
        # If node contains "earnings per share" in its text, print the node

        if "earnings per share" in node.text.lower():
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


 
    
if __name__ == "__main__":
    # Extract the HTML content of the filing\
    # Download the latest 10-Q filing for Apple
    query = "AAPL/0000320193-24-000069"
    dl = Downloader(config.COMPANY_NAME, config.EMAIL)
    html_content = dl.get_filing_html(query=query)

    # Extract relevant EPS data from the HTML content
    eps_text_data = extract_relevant_eps_data_html(query=query)

    if eps_text_data:
        # Extract EPS values using OpenAI
        eps_json = extract_eps_data(eps_text_data)
        data = extract_eps_from_openai_result(eps_json)
        basic_eps, diluted_eps = data
        print(f"Basic EPS: {basic_eps}, Diluted EPS: {diluted_eps}")
    else:
        print("No relevant EPS data found in the filing.")

    


