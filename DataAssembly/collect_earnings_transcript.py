import os
import sys
# === Load config and setup ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import platform
import pandas as pd
import sqlite3

def sec_search(ticker_list):
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.sec.gov/edgar/search/")
    wait = WebDriverWait(driver, 10)
    ticker = ticker_list[0]
    if len(ticker_list) > 1:
        ticker_list = ticker_list[1:]
    # Input ticker
    input_field = wait.until(EC.element_to_be_clickable((By.ID, "entity-short-form")))
    input_field.click()
    for char in ticker:
        input_field.send_keys(char)
        time.sleep(0.1)

    time.sleep(2)  # Let dropdown load
    typed_value = input_field.get_attribute("value")
    print(f"Typed into box: '{typed_value}'")

    # Select top dropdown via keyboard (this worked before)
    input_field.send_keys(Keys.ARROW_DOWN)
    input_field.send_keys(Keys.RETURN)
    print(f"✅ Selected ticker via keyboard: {ticker}")

    # Wait for results to load
    wait.until(lambda d: f"{ticker}" in d.current_url and "action" in d.page_source.lower())
    print("✅ Likely on search results page.")


    time.sleep(2)  # Allow results to stabilize

    # Clear and set date inputs
    start_input = wait.until(EC.element_to_be_clickable((By.ID, "date-from")))
    end_input = wait.until(EC.element_to_be_clickable((By.ID, "date-to")))
    mod = Keys.COMMAND if platform.system() == "Darwin" else Keys.CONTROL

    # Clear and input start date
    start_input.click()
    start_input.send_keys(mod + "a")
    start_input.send_keys(Keys.BACKSPACE)
    time.sleep(0.3)
    start_input.send_keys("01-01-2012")

    # Clear and input end date
    end_input.click()
    end_input.send_keys(mod + "a")
    end_input.send_keys(Keys.BACKSPACE)
    time.sleep(0.3)
    end_input.send_keys("12-31-2024")

    print("✅ Date range set: 2012-01-01 to 2024-12-31")

    time.sleep(1)


    # Open the "Form" filter dropdown explicitly
    form_section = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@data-toggle='collapse' and contains(@href, '#collapseTwo2')]")))
    form_section.click()
    time.sleep(1)  # Give time for the dropdown animation
    form_section.click()

    # Now click the "10-Q" form filter
    tenq_filter = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-filter-key="10-Q"]')))
    driver.execute_script("arguments[0].scrollIntoView(true);", tenq_filter)
    tenq_filter.click()
    print("✅ Filtered for 10-Q forms")

    time.sleep(1)  # Allow inputs to register

    all_dfs = []
    rows = driver.find_elements(By.CSS_SELECTOR, "tr")

    results = []

    for row in rows:
        try:
            link = row.find_element(By.CSS_SELECTOR, "a.preview-file")
            if "10-Q" in link.text:
                url = link.get_attribute("href")
                tds = row.find_elements(By.TAG_NAME, "td")
                filed_date = tds[1].text if len(tds) > 1 else ""
                reporting_date = tds[2].text if len(tds) > 2 else ""
                results.append({
                    "Ticker": ticker,
                    "Form Type": link.text,
                    "Filed Date": filed_date,
                    "Reporting Date": reporting_date,
                    "URL": url
                })
        except Exception:
            continue
    print(f"✅ Found {len(results)} 10-Q filings for {ticker}.")
    
    all_dfs.append(pd.DataFrame(results))

    if len(ticker_list) == 1:
        driver.quit()
        return all_dfs

    if ticker_list:
        print(f"Processing next ticker: {ticker_list[0]}")

        for index,ticker in enumerate(ticker_list):
            print(f"Processing ticker {index + 1}/{len(ticker_list)}: {ticker}")
            try:
                if len(all_dfs)% 10 == 0:
                    print("Saving progress to CSV...")
                    if all_dfs:
                        final_df = pd.concat(all_dfs, ignore_index=True)
                        final_df.to_csv(config.FILING_DATES_AND_URLS_CSV, index=False)
                # Click on the search box, clear it, and input the next ticker
                input_field = wait.until(EC.element_to_be_clickable((By.ID, "entity-full-form")))
                input_field.click()
                input_field.send_keys(Keys.COMMAND + "a")  # Clear the input
                input_field.send_keys(Keys.BACKSPACE)  # Clear the input
                time.sleep(0.5)  # Allow time for the input to clear
                for char in ticker:
                    input_field.send_keys(char)
                    time.sleep(0.1)
                print(f"✅ Input ticker: {ticker}")
                time.sleep(1)  # Let dropdown load
                typed_value = input_field.get_attribute("value")
                print(f"Typed into box: '{typed_value}'")
                # Select top dropdown via keyboard (this worked before)
                input_field.send_keys(Keys.ARROW_DOWN)
                input_field.send_keys(Keys.RETURN)
                print(f"✅ Selected ticker via keyboard: {ticker}")
                # Wait for results to load
                wait.until(lambda d: f"{ticker}" in d.current_url and "action" in d.page_source.lower())
                print("✅ Likely on search results page.")
                time.sleep(1)  # Allow results to stabilize
                # Dates should already be set, so we don't need to set them again
                # Open the "Form" filter dropdown explicitly
                form_section = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@data-toggle='collapse' and contains(@href, '#collapseTwo2')]")))
                form_section.click()
                time.sleep(1)  # Give time for the dropdown animation
                form_section.click()
                # Now click the "10-Q" form filter
                tenq_filter = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-filter-key="10-Q"]')))
                driver.execute_script("arguments[0].scrollIntoView(true);", tenq_filter)
                tenq_filter.click()
                print(f"✅ Filtered for 10-Q forms for {ticker}")
                time.sleep(1)  # Allow inputs to register
                # Collect results for the current ticker
                rows = driver.find_elements(By.CSS_SELECTOR, "tr")
                results = []
                results = []

                for row in rows:
                    try:
                        link = row.find_element(By.CSS_SELECTOR, "a.preview-file")
                        if "10-Q" in link.text:
                            url = link.get_attribute("href")
                            tds = row.find_elements(By.TAG_NAME, "td")
                            filed_date = tds[1].text if len(tds) > 1 else ""
                            reporting_date = tds[2].text if len(tds) > 2 else ""
                            results.append({
                                "Ticker": ticker,
                                "Form Type": link.text,
                                "Filed Date": filed_date,
                                "Reporting Date": reporting_date,
                                "URL": url
                            })
                    except Exception:
                        continue
                print(f"✅ Found {len(results)} 10-Q filings for {ticker}.")
                
                all_dfs.append(pd.DataFrame(results))
            
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")
                continue

    driver.quit()
    return all_dfs
                


def generate_from_scratch():

    # Open the database and get the unique tickers from the metadata CSV
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT ticker FROM {config.PRICING_TABLE_NAME}")
    tickers = [row[0] for row in cursor.fetchall()]
    print(tickers)
    conn.close()
    print(f"Found {len(tickers)} unique tickers in the database.")

    all_dfs = None

    try:
        all_dfs = sec_search(tickers)
    except Exception as e:
        print(f"Error during SEC search: {e}")


    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(config.FILING_DATES_AND_URLS_CSV, index=False)

def add_to_existing(ticker):

    """
    Add a single ticker's 10-Q filings to the existing CSV.
    """
    try:
        results = sec_search(ticker)
        if not results.empty:
            # concatenate with existing CSV
            if os.path.exists(config.FILING_DATES_AND_URLS_CSV):
                existing_df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
                results = pd.concat([existing_df, results[0]], ignore_index=True)
            results.to_csv(config.FILING_DATES_AND_URLS_CSV, index=False)
        else:
            print(f"No 10-Q filings found for {ticker}.")
    except Exception as e:
        print(f"Error adding {ticker}: {e}")

def add_missing_tickers():

    """
    This adds collects the filings and filing dates for tickers that are in the database but not in the existing CSV.
    """
    # Go through unique tickers in config.DB_PATH and see which are missing in config.FILING_DATES_AND_URLS_CSV
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT ticker FROM {config.PRICING_TABLE_NAME}")
    existing_tickers = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Make sure the CSV exists
    if not os.path.exists(config.FILING_DATES_AND_URLS_CSV):
        print(f"CSV file {config.FILING_DATES_AND_URLS_CSV} does not exist. Generating from scratch.")
        generate_from_scratch()
        return
    
    existing_df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
    existing_tickers_in_csv = existing_df['Ticker'].unique().tolist()
    missing_tickers = set(existing_tickers) - set(existing_tickers_in_csv)
    print(f"Found {len(missing_tickers)} missing tickers to add from the database.")
    for ticker in missing_tickers:
        print(f"Adding missing ticker: {ticker}")
        add_to_existing(ticker)






if __name__ == "__main__":

    add_missing_tickers()



    


