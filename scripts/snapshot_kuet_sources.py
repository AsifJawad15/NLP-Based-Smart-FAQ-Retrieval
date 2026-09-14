"""Archive public KUET source text and links for a dated manual FAQ review."""
import argparse
import json
import urllib.request
from urllib.parse import urlparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

parser = argparse.ArgumentParser()
parser.add_argument('paths', nargs='+')
args = parser.parse_args()
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
out = Path(__file__).resolve().parents[1] / 'data/kuet/source_snapshots' / date.today().isoformat()
out.mkdir(parents=True, exist_ok=True)

def fetch(path):
    url = path if path.startswith('https://') else 'https://www.kuet.ac.bd/' + path
    parsed = urlparse(url)
    if parsed.hostname not in {'www.kuet.ac.bd', 'www2.kuet.ac.bd', 'ossbhtpa.gov.bd', 'www.kbec-official.org', 'admission.kuet.ac.bd'}:
        return path, 'FETCH FAILED: Source host is not in the review allowlist'
    try:
        with urllib.request.urlopen(url, timeout=40) as response:
            raw = response.read()
            final_url = response.url
        soup = BeautifulSoup(raw, 'html.parser')
        links = [{'label': a.get_text(' ', strip=True), 'url': a.get('href')} for a in soup.select('a[href]')]
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        text = soup.get_text(' ', strip=True)
        name = (parsed.hostname + parsed.path).replace('/', '_') if path.startswith('https://') else path.replace('/', '_') or 'home'
        (out / (name + '.txt')).write_text(text, encoding='utf-8')
        (out / (name + '.json')).write_text(json.dumps({'url': url, 'final_url': final_url, 'retrieved_on': date.today().isoformat(), 'links': links}, indent=2), encoding='utf-8')
        return path, text
    except Exception as error:
        return path, 'FETCH FAILED: ' + str(error)

if __name__ == '__main__':
    for path, text in ThreadPoolExecutor(max_workers=4).map(fetch, args.paths):
        print('\nPAGE', path, '\n', text[:250])
