def creates_tuple():
    return (1, 2, 4)


def takes_tuple_returns_int(input_tuple):
    return sum(input_tuple)


def takes_tuples(tuples):
    return [sum(t) for t in tuples]
