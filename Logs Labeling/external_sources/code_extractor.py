import json
import os
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
INPUT_DIR = "data/mitre_data"   # Reads from the JSONs created in Step 1
OUTPUT_DIR = "data/cti_code_only" # Outputs to this new folder
TIMEOUT = 15
MAX_WORKERS = 10 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def extract_text_from_pdf(content_bytes):
    """
    PDFs don't have <code> tags. We return full text or a placeholder.
    """
    try:
        reader = PdfReader(BytesIO(content_bytes))
        text = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text.append(t)
        return "PDF_CONTENT_FULL:\n" + "\n".join(text)
    except Exception:
        return None

def fetch_code_blocks(url):
    """
    Fetches URL and extracts ONLY content inside <pre> and <code> tags.
    """
    if not url or "mitre.org" in url:
        return url, None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
        resp.raise_for_status()
        
        content_type = resp.headers.get('Content-Type', '').lower()
        
        # 1. Handle PDF (Cannot filter for code tags)
        if 'application/pdf' in content_type or url.endswith('.pdf'):
            text = extract_text_from_pdf(resp.content)
            return url, text

        # 2. Handle HTML (Filter for Code)
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        extracted_blocks = []
        
        # We look for 'pre' (blocks) and 'code' (inline/blocks)
        # We also check for 'textarea' which sometimes holds copy-paste code
        targets = soup.find_all(['pre', 'code', 'textarea'])
        
        for tag in targets:
            # OPTIONAL: Ignore <code> if it is inside a <pre> to avoid duplicates
            # (BeautifulSoup finds both, but <pre> usually contains the <code>)
            if tag.name == 'code' and tag.find_parent('pre'):
                continue
            
            text = tag.get_text(strip=True)
            
            # Filter out empty or extremely short snippets
            if len(text) > 2: 
                extracted_blocks.append(text)
        
        if extracted_blocks:
            # Join them with a separator
            final_output = "\n".join(extracted_blocks)
            return url, final_output
        else:
            return url, None # No code blocks found

    except Exception as e:
        return url, None

def process_file(filepath):
    filename = os.path.basename(filepath)
    print(f"[*] Processing {filename}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Collect Unique URLs
    unique_urls = set()
    for tech in data.get('techniques', []):
        for proc in tech.get('procedure_examples', []):
            for ref_url in proc.get('reference_urls', []):
                if "mitre.org" not in ref_url:
                    unique_urls.add(ref_url)

    print(f"    - Found {len(unique_urls)} unique external references.")

    # 2. Fetch Code Blocks
    url_cache = {} 
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(fetch_code_blocks, url): url for url in unique_urls}
        
        for future in as_completed(future_to_url):
            url, text = future.result()
            if text:
                url_cache[url] = text
    
    print(f"    - Found code blocks in {len(url_cache)} URLs.")

    # 3. Reconstruct JSON
    new_tactic_data = {
        "tactic_id": data.get("tactic_id"),
        "tactic_name": data.get("tactic_name"),
        "techniques": []
    }

    for tech in data.get('techniques', []):
        new_tech = {
            "technique_id": tech['technique_id'],
            # RECORDING NAME: This passes the corrected name from Step 1 through to the output
            "technique_name": tech['technique_name'], 
            "procedure_examples": []
        }

        for proc in tech.get('procedure_examples', []):
            new_proc = {
                "procedure_id": proc.get('procedure_id'),
                "name": proc.get('name'),
                "description": proc.get('description'),
                "code_snippets": [] 
            }

            for ref_url in proc.get('reference_urls', []):
                content = url_cache.get(ref_url)
                if content:
                    new_proc['code_snippets'].append({
                        "url": ref_url,
                        "code_content": content
                    })
            
            # Only append procedure if we actually found code snippets
            if new_proc['code_snippets']:
                new_tech['procedure_examples'].append(new_proc)

        if new_tech['procedure_examples']:
            new_tactic_data['techniques'].append(new_tech)

    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_tactic_data, f, indent=4, ensure_ascii=False)
    
    print(f"    + Saved to {output_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            process_file(os.path.join(INPUT_DIR, filename))

if __name__ == "__main__":
    main()