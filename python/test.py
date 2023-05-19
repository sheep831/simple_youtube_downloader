import math
import string
def cipher_text(plain_text):
    text = sanitize_text(plain_text)
    print(text)
    
    # Exit early if we have an empty string, as we
    # don't want a divide by zero error.
    if len(text) == 0:
        return ''
    
    # math.ceil to round up always. It is better to have
    # too many cells instead of too few.
    cols = math.ceil(math.sqrt(len(text)))
    rows = math.ceil(len(text) / cols)
    # add spaces to bring the string to cols*rows characters
    text += ' ' * ((cols * rows) - len(text))
    # For each column make a string. Each string is a character from
    # the string skipping column characters for each step.
    return ' '.join(text[i::cols] for i in range(cols))

def sanitize_text(plain_text):
    trantab = str.maketrans('', '', string.punctuation + string.whitespace)
    return plain_text.lower().strip().translate(trantab)

value = "Chill out."    
lines = "If man was meant to stay on the ground, god would have given us roots."
print(sanitize_text(lines))
print(sanitize_text(value))

