import re
import math

def organize_into_rectangle(message_length):
    # Find the square root of the message length
    square_root = math.isqrt(message_length)

    # Calculate the number of columns and rows
    if square_root * square_root >= message_length:
        c = square_root
        r = square_root
    elif square_root * (square_root + 1) >= message_length:
        c = square_root + 1
        r = square_root 
    else:
        c = square_root + 1
        r = square_root + 1

    return r, c

if __name__ == "__main__":
    def cipher_text(plain_text):
        text = re.sub(r'[^a-zA-Z]', '', plain_text).lower()
        rc = organize_into_rectangle(len(text))
        result = ""
        for i in range(rc[1]):
            units = text[i::rc[1]]
            if len(units) < rc[0]:
                units += " " * (rc[0] - len(units))
            result += units + " "
        return result
    value = "Chill out."    
    lines = "If man was meant to stay on the ground, god would have given us roots."
    print(cipher_text(lines))
    print(cipher_text(value)) # "clu hlt io "