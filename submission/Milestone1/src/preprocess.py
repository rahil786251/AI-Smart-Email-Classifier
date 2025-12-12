import re
from bs4 import BeautifulSoup

# -------------------------------------------------------------
# 1) Remove HTML and extract visible text
# -------------------------------------------------------------
def html_to_text(html):
    if not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style tags completely
    for s in soup(["script", "style"]):
        s.decompose()

    return soup.get_text(separator="\n")


# -------------------------------------------------------------
# 2) Remove quoted replies (“On Jan 2, John wrote:”)
#    or lines starting with ">"
# -------------------------------------------------------------
def remove_quoted_replies(text):
    if not isinstance(text, str):
        return ""

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        if line.strip().startswith(">"):
            continue
        
        # Stop at reply header
        if re.match(r'^\s*On .* wrote:', line):
            break

        cleaned.append(line)

    return "\n".join(cleaned).strip()


# -------------------------------------------------------------
# 3) Remove common email signatures
# -------------------------------------------------------------
SIGNATURE_PATTERNS = [
    r'^--\s*$',                    # --
    r'^thanks[,]?$',               # Thanks
    r'^regards[,]?$',              # Regards
    r'^best[,]?$',                 # Best
    r'^cheers[,]?$',               # Cheers
    r'^sincerely[,]?$',            # Sincerely
    r'^yours[,]?$',                # Yours
    r'^sent from my (iphone|android)',
]

SIGNATURE_RE = re.compile("|".join(SIGNATURE_PATTERNS), re.IGNORECASE | re.MULTILINE)


def remove_signatures(text):
    if not isinstance(text, str):
        return ""
    match = SIGNATURE_RE.search(text)
    if match:
        return text[:match.start()].strip()
    return text


# -------------------------------------------------------------
# 4) Remove URLs, email IDs, extra whitespace
# -------------------------------------------------------------
URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')

def clean_noise(text):
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------------------------------------------------------------
# 5) MASTER FUNCTION
# -------------------------------------------------------------
def clean_email(text):
    if not isinstance(text, str):
        return ""

    text = html_to_text(text)
    text = remove_quoted_replies(text)
    text = remove_signatures(text)
    text = clean_noise(text)

    return text
