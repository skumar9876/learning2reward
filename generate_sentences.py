import itertools

'''
Generates all possible binary vectors of size len
representing all possible sentences

Returns 2D array
'''
def generate_sentences(len):
    strings = [''.join(x) for x in itertools.product('01', repeat=len)]
    arrays = []
    for string in strings:
        vector = []
        for letter in string:
            vector.append(int(letter))
        arrays.append(vector)
    return arrays