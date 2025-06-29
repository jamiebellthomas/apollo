import os
import sys
# === Load config and setup ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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

    all_dfs = []

    for i, ticker in enumerate(ticker_list):
        print(f"🔍 Processing ticker {i + 1}/{len(ticker_list)}: {ticker}")
        try:
            input_id = "entity-short-form" if i == 0 else "entity-full-form"
            input_field = wait.until(EC.element_to_be_clickable((By.ID, input_id)))
            input_field.click()
            input_field.send_keys(Keys.COMMAND + "a")
            input_field.send_keys(Keys.BACKSPACE)
            time.sleep(0.5)

            for char in ticker:
                input_field.send_keys(char)
                time.sleep(0.1)

            time.sleep(2)
            input_field.send_keys(Keys.ARROW_DOWN)
            input_field.send_keys(Keys.RETURN)

            wait.until(lambda d: f"{ticker}" in d.current_url and "action" in d.page_source.lower())
            time.sleep(2)

            if i == 0:
                start_input = wait.until(EC.element_to_be_clickable((By.ID, "date-from")))
                end_input = wait.until(EC.element_to_be_clickable((By.ID, "date-to")))
                mod = Keys.COMMAND if platform.system() == "Darwin" else Keys.CONTROL

                start_input.click()
                start_input.send_keys(mod + "a")
                start_input.send_keys(Keys.BACKSPACE)
                time.sleep(0.3)
                start_input.send_keys("01-01-2012")

                end_input.click()
                end_input.send_keys(mod + "a")
                end_input.send_keys(Keys.BACKSPACE)
                time.sleep(0.3)
                end_input.send_keys("12-31-2024")
                print("✅ Date range set.")

            
            # --- 1. Collect 10-Qs ---

            form_section = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@data-toggle='collapse' and contains(@href, '#collapseTwo2')]")))
            form_section.click()
            time.sleep(1)
            form_section.click()

            tenq_filter = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-filter-key="10-Q"]')))
            driver.execute_script("arguments[0].scrollIntoView(true);", tenq_filter)
            tenq_filter.click()
            print("✅ Filtered for 10-Q forms.")

            time.sleep(1)
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

            print(f"✅ Found {len(results)} 10-Q filings for {ticker}")
            all_dfs.append(pd.DataFrame(results))

            # --- 2. Clear 10-Q filter ---
            clear_10q = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-button="10-Q"]')))
            clear_10q.click()
            time.sleep(1)

            # --- 3. Collect 10-Ks ---
            form_section = wait.until(EC.element_to_be_clickable((By.XPATH, "//a[@data-toggle='collapse' and contains(@href, '#collapseTwo2')]")))
            form_section.click()
            time.sleep(1)
            form_section.click()
            tenk_filter = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[data-filter-key="10-K"]')))
            driver.execute_script("arguments[0].scrollIntoView(true);", tenk_filter)
            tenk_filter.click()
            print("✅ Filtered for 10-K forms.")
            time.sleep(1)
            rows = driver.find_elements(By.CSS_SELECTOR, "tr")
            results = []
            for row in rows:
                try:
                    link = row.find_element(By.CSS_SELECTOR, "a.preview-file")
                    if "10-K" in link.text:
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
            print(f"✅ Found {len(results)} 10-K filings for {ticker}")
            all_dfs.append(pd.DataFrame(results))
            # --- 4. Clear 10-K filter ---
            clear_10k = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-button="10-K"]')))
            clear_10k.click()
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
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

def add_to_existing(ticker_list):

    """
    Add a single ticker's 10-Q filings to the existing CSV.
    """
    try:
        results = sec_search(ticker_list)
        if len(results) != 0:
            # Read the existing CSV
            existing_df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
            # results is a list of DataFrames, so we need to concatenate them
            results = pd.concat(results, ignore_index=True)

            # Concatenate the new results with the existing DataFrame
            updated_df = pd.concat([existing_df, results], ignore_index=True)
            # Write the updated DataFrame back to the CSV
            updated_df.to_csv(config.FILING_DATES_AND_URLS_CSV, index=False)
            print(f"✅ Successfully added {len(results)} filings to the existing CSV.")

    except Exception as e:
        print(f"Error adding data: {e}")

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
    # convert set back to list 
    missing_tickers = list(missing_tickers)
    if len(missing_tickers) == 0:
        print("[INFO] No missing tickers found. All tickers in the database are already in the CSV.")
        return
    add_to_existing(missing_tickers)






if __name__ == "__main__":

    add_missing_tickers()



    


