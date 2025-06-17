import requests
from bs4 import BeautifulSoup

def save_parsed_10q_to_txt(index_url, output_path="parsed_10q.txt"):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Encoding": "gzip, deflate"
    }

    try:
        # Step 1: Get the index page
        response = requests.get(index_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Step 2: Find the first .xml or .xbrl link (XBRL instance document)
        xbrl_link = None
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if href.endswith(".xml") or href.endswith(".xbrl"):
                xbrl_link = href
                break

        if not xbrl_link:
            print("❌ No XBRL document found.")
            return None

        # Step 3: Build full URL to the XBRL file
        base_url = index_url.rsplit("/", 1)[0]
        full_xbrl_url = f"{base_url}/{xbrl_link}"

        # Step 4: Download and parse XBRL
        xbrl_response = requests.get(full_xbrl_url, headers=headers)
        xbrl_response.raise_for_status()
        xbrl_soup = BeautifulSoup(xbrl_response.content, "lxml-xml")  # Parse as XML

        # Step 5: Save parsed XML to text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xbrl_soup.prettify())

        print(f"✅ Parsed XBRL saved to {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    

if __name__ == "__main__":
    index_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm"
    output_path = "parsed_10q.txt"
    save_parsed_10q_to_txt(index_url, output_path)
