import re 

def clean_text(text: str) -> str:
    # Handle None or empty input
    if not text:
        return ""
    
    # Replace Unicode escapes (like \u2026)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove newlines and carriage returns
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Dictionary of Unicode characters to replace with their ASCII equivalents
    unicode_map = {
        '\u2019': "'",    # right single quotation mark
        '\u2018': "'",    # left single quotation mark
        '\u201c': '"',    # left double quotation mark
        '\u201d': '"',    # right double quotation mark
        '\u2026': "...",  # horizontal ellipsis
        '\u2013': "-",    # en dash
        '\u2014': "--",   # em dash
        '\u00a0': " ",    # non-breaking space
        '\u200b': "",     # zero-width space
        '\u200e': "",     # left-to-right mark
        '\u200f': "",     # right-to-left mark
        '\xa0': " ",      # non-breaking space
        '\n': " ",        # newline
        '\r': " ",        # carriage return
        '\t': " ",        # tab
        '\f': " ",        # form feed
        '\v': " "         # vertical tab
    }
    
     # Second pass: handle escaped quotes
    # text = text.replace('\\"', '"')
    text = text.replace('"', "")
    text = text.replace("'", "")
    
    # Third pass: standardize any remaining quotes
    text = re.sub(r'(?<!\\)"', '"', text)  # handle unescaped double quotes
    
    # Remove control characters while preserving spaces
    text = ''.join(char for char in text if ord(char) >= 32 or char == ' ')
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()