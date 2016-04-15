#Chunk list in chunks of size n
def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]

#Split list into n parts
def split(lst,n):
    return [ lst[i::n] for i in range(n if n < len(lst) else len(lst)) ]
