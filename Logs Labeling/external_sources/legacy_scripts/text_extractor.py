import json
import os
import requests
import trafilatura
from io import BytesIO
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# Disable SSL warnings for cleaner output (common with old CTI blogs)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
INPUT_DIR = "data/mitre_data"
OUTPUT_DIR = "data/cti_reports"
TIMEOUT = 18
MAX_WORKERS = 24 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def extract_text_from_pdf(content_bytes):
    """Extracts text from PDF bytes."""
    try:
        reader = PdfReader(BytesIO(content_bytes))
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "\n".join(text)
    except Exception:
        return None

def fetch_and_extract(url):
    """
    Downloads URL and extracts main content.
    Returns: (url, content_text)
    """
    if not url:
        return url, None

    # Skip MITRE.org references
    if "mitre.org" in url:
        return url, None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
        resp.raise_for_status()
        
        content_type = resp.headers.get('Content-Type', '').lower()
        
        # Handle PDF
        if 'application/pdf' in content_type or url.endswith('.pdf'):
            text = extract_text_from_pdf(resp.content)
            return url, text

        # Handle HTML
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        return url, text

    except Exception as e:
        # print(f"Failed: {url} | {e}") # Uncomment for debug
        return url, None

def process_file(filepath):
    filename = os.path.basename(filepath)
    print(f"[*] Processing {filename}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Collect all UNIQUE URLs first (to avoid downloading same report 50 times)
    unique_urls = set()
    for tech in data.get('techniques', []):
        for proc in tech.get('procedure_examples', []):
            for ref_url in proc.get('reference_urls', []):
                if "mitre.org" not in ref_url:
                    unique_urls.add(ref_url)

    print(f"    - Found {len(unique_urls)} unique external references.")

    # 2. Fetch URLs concurrently
    url_cache = {} # Map: URL -> Extracted Text
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(fetch_and_extract, url): url for url in unique_urls}
        
        count = 0
        total = len(unique_urls)
        
        for future in as_completed(future_to_url):
            url, text = future.result()
            if text and len(text) > 100: # Filter out empty/too short results
                url_cache[url] = text
            
            count += 1
            if count % 10 == 0:
                print(f"    - Progress: {count}/{total} URLs processed", end="\r")
    
    print(f"\n    - Successfully extracted text from {len(url_cache)} URLs.")

    # 3. Reconstruct Data Structure (Injecting Content)
    # This creates the duplicates you asked for, organizing by Technique -> Procedure
    new_tactic_data = {
        "tactic_id": data.get("tactic_id"),
        "tactic_name": data.get("tactic_name"),
        "techniques": []
    }

    for tech in data.get('techniques', []):
        new_tech = {
            "technique_id": tech['technique_id'],
            "technique_name": tech['technique_name'],
            "procedure_examples": []
        }

        for proc in tech.get('procedure_examples', []):
            new_proc = {
                "procedure_id": proc.get('procedure_id'),
                "name": proc.get('name'),
                "description": proc.get('description'),
                "references_data": [] # Changed from reference_urls to list of objects
            }

            for ref_url in proc.get('reference_urls', []):
                # Skip mitre
                if "mitre.org" in ref_url:
                    continue
                
                # Check cache
                content = url_cache.get(ref_url)
                
                # Only add if we actually got content
                if content:
                    new_proc['references_data'].append({
                        "url": ref_url,
                        "content": content
                    })
            
            # Only add the procedure if we found valid reference data
            if new_proc['references_data']:
                new_tech['procedure_examples'].append(new_proc)

        if new_tech['procedure_examples']:
            new_tactic_data['techniques'].append(new_tech)

    # 4. Save
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_tactic_data, f, indent=4, ensure_ascii=False)
    
    print(f"    + Saved populated JSON to {output_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            process_file(os.path.join(INPUT_DIR, filename))

if __name__ == "__main__":
    main()