import re, collections
def load(p):
    s = open(p).read(); s = re.sub(r'(?<!\\)%.*', '', s); return s.replace('{,}', '.')
en, es = load('../manuscrito_v3_en.tex'), load('../manuscrito_v3_es.tex')
def maths(s):
    out = []
    out += re.findall(r'(?<!\\)\$([^$]+)\$', s)
    out += re.findall(r'\\\[(.+?)\\\]', s, re.S)
    for env in ['align\*?', 'gather\*?', 'multline\*?', 'equation\*?']:
        out += re.findall(r'\\begin\{(?:%s)\}(.+?)\\end\{(?:%s)\}' % (env, env), s, re.S)
    norm = []
    for m in out:
        m = re.sub(r'\\(?:text|mathrm|textrm|emph|textit)\{[^{}]*\}', 'T', m)   # drop words
        m = re.sub(r'\\label\{[^}]*\}', '', m)
        m = re.sub(r'\s+', '', m)
        norm.append(m)
    return collections.Counter(norm)
a, b = maths(en), maths(es)
print('math segments EN', sum(a.values()), 'ES', sum(b.values()))
d1, d2 = a-b, b-a
print('--- only EN'); [print(repr(k)) for k in d1]
print('--- only ES'); [print(repr(k)) for k in d2]
