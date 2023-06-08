def find_anagrams(word, candidates):
    result = []
    word = list(word.lower())
    word_transform = sorted(word)
    for candidate in candidates:
        if word == list(candidate.lower()):
            continue
        candidate_transform = list(candidate.lower())
        candidate_transform.sort()
        if candidate_transform == word_transform:
            result.append(candidate)
    return result




candidates = ["Banana"]
print(find_anagrams("BANANA", candidates))