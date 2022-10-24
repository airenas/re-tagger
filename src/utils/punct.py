punctuations = '…)`»~„-$™[•<=]%‘,>\“”§±—+·’;&:×–?!°.*\'("/'


def is_punctuation(l1):
    return l1 in punctuations or l1 == "..."
