import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import json
import os
import re

# Configuration
BASE_URL = "https://attack.mitre.org"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# The full list of Enterprise Tactics
TACTICS_LIST = [
    {"id": "TA0043", "name": "Reconnaissance"},
    {"id": "TA0042", "name": "Resource Development"},
    {"id": "TA0001", "name": "Initial Access"},
    {"id": "TA0002", "name": "Execution"},
    {"id": "TA0003", "name": "Persistence"},
    {"id": "TA0004", "name": "Privilege Escalation"},
    {"id": "TA0005", "name": "Defense Evasion"},
    {"id": "TA0006", "name": "Credential Access"},
    {"id": "TA0007", "name": "Discovery"},
    {"id": "TA0008", "name": "Lateral Movement"},
    {"id": "TA0009", "name": "Collection"},
    {"id": "TA0011", "name": "Command and Control"},
    {"id": "TA0010", "name": "Exfiltration"},
    {"id": "TA0040", "name": "Impact"}
]

def get_soup(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching {url}: {e}")
        return None

def format_id_from_url(href):
    """Parses URL to get T1595 or T1595.001"""
    parts = [p for p in href.split('/') if p]
    if len(parts) >= 3 and parts[-1].isdigit():
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]

def get_techniques_from_tactic(tactic_id):
    """Parses a tactic page to find all techniques."""
    url = f"{BASE_URL}/tactics/{tactic_id}/"
    soup = get_soup(url)
    if not soup:
        return []

    techniques = []
    tech_table = soup.find('table', class_='table-techniques')
    
    if tech_table:
        rows = tech_table.find_all('tr')
        for row in rows:
            link_tag = row.find('a')
            if link_tag:
                href = link_tag['href']
                name = link_tag.get_text(strip=True)
                
                if "/techniques/T" in href:
                    full_url = urllib.parse.urljoin(BASE_URL, href)
                    full_id = format_id_from_url(href)
                    
                    techniques.append({
                        'id': full_id,
                        'name': name,
                        'url': full_url
                    })
    return techniques

def get_procedure_examples(technique_url):
    """
    Parses a technique page to extract Procedure Examples.
    Also extracts the specific URL for the Group/Software.
    """
    soup = get_soup(technique_url)
    if not soup:
        return []

    procedures = []
    
    # Find Header by text "Procedure Examples"
    header = soup.find(lambda tag: tag.name == "h2" and "Procedure Examples" in tag.get_text())
    
    if header:
        table = header.find_next('table')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    p_id = cols[0].get_text(strip=True)
                    p_name = cols[1].get_text(strip=True)
                    p_desc = cols[2].get_text(" ", strip=True)
                    
                    # EXTRACT URL for the specific procedure (Group/Software)
                    # The link is usually in the 2nd column (Name)
                    link_tag = cols[1].find('a')
                    related_url = None
                    if link_tag and link_tag.get('href'):
                        related_url = urllib.parse.urljoin(BASE_URL, link_tag['href'])

                    if p_id == "ID" and p_name == "Name": continue

                    procedures.append({
                        'procedure_id': p_id,
                        'name': p_name,
                        'related_url': related_url, # URL to the specific Group/Software
                        'description': p_desc
                    })
    return procedures

def sanitize_filename(name):
    """Makes a string safe for filenames."""
    return re.sub(r'[^\w\-_\. ]', '_', name)

def main():
    # Create a directory to store the files
    output_dir = "mitre_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"[*] Starting Crawl of {len(TACTICS_LIST)} Tactics...")
    print(f"[*] Output directory: ./{output_dir}/")

    for tactic in TACTICS_LIST:
        t_id = tactic['id']
        t_name = tactic['name']
        print(f"\n[{t_id}] Crawling Tactic: {t_name}")
        
        # 1. Get all techniques for this tactic
        techniques = get_techniques_from_tactic(t_id)
        print(f"    Found {len(techniques)} techniques. Extracting procedures...")
        
        tactic_data = {
            "tactic_id": t_id,
            "tactic_name": t_name,
            "techniques": []
        }

        # 2. Loop through every technique/sub-technique
        for tech in techniques:
            # Politeness delay
            time.sleep(0.2) 
            
            examples = get_procedure_examples(tech['url'])
            
            tech_entry = {
                "technique_id": tech['id'],
                "technique_name": tech['name'],
                "technique_url": tech['url'],
                "procedure_examples": examples
            }
            tactic_data["techniques"].append(tech_entry)
            
            # Print a dot for progress
            print(".", end="", flush=True)

        # 3. Save to a separate JSON file for this Tactic
        filename = f"{t_id}_{sanitize_filename(t_name)}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tactic_data, f, indent=4, ensure_ascii=False)
            
        print(f"\n    Saved {len(tactic_data['techniques'])} techniques to {filename}")

    print("\n[*] Full Crawl Complete.")

if __name__ == "__main__":
    main()