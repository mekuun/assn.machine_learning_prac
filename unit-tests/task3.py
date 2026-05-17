def find_modified_max_argmax(L, f):
    j = 0
    m = []
    for i in L:
        if type(i) == int:
            m.append((f(i), j))
            j += 1
    return (max(m)[0], max(m)[1]) if m else ()