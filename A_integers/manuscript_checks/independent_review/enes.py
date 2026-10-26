import re, collections, sys
en = open('../manuscrito_v3_en.tex').read(); es = open('../manuscrito_v3_es.tex').read()
def strip_comments(s): return re.sub(r'(?<!\\)%.*', '', s)
en, es = strip_comments(en), strip_comments(es)
lab = lambda s: re.findall(r'\\label\{([^}]*)\}', s)
print('labels equal order:', lab(en) == lab(es), len(lab(en)), len(lab(es)))
for kind in ['cite', 'ref', 'eqref']:
    f = lambda s: collections.Counter(x for m in re.findall(r'\\%s(?:\[[^\]]*\])?\{([^}]*)\}' % kind, s) for x in m.split(','))
    a, b = f(en), f(es)
    print(kind, 'equal:', a == b, 'EN-ES:', dict(a-b), 'ES-EN:', dict(b-a))
envs = lambda s: collections.Counter(re.findall(r'\\begin\{(theorem|proposition|lemma|corollary|claim|conjecture|hypothesis|definition|remark|proof|table)\}', s))
print('envs equal:', envs(en) == envs(es), envs(en))
# decimals: normalise Spanish decimal comma {,}
def nums(s):
    s = s.replace('{,}', '.')
    return collections.Counter(re.findall(r'(?<![\w.])-?\d+\.\d+', s))
a, b = nums(en), nums(es)
print('decimals equal:', a == b, 'EN-ES:', dict(a-b), 'ES-EN:', dict(b-a))
# integers with >=2 digits in math
ints = lambda s: collections.Counter(re.findall(r'(?<![\w.{,])\d{2,}(?![\w.])', s.replace('{,}', '.')))
a, b = ints(en), ints(es)
print('ints EN-ES:', dict(a-b)); print('ints ES-EN:', dict(b-a))
st = lambda s: len(re.findall(r'\\st\{', s))
print('status lines', st(en), st(es))
# sequence of environment+label pairs
seq = lambda s: re.findall(r'\\begin\{(theorem|proposition|lemma|corollary|claim|conjecture|hypothesis|definition|remark)\}(?:\[[^\]]*\])?\s*(?:\\label\{([^}]*)\})?', s)
print('env/label sequence equal:', seq(en) == seq(es))
for x, y in zip(seq(en), seq(es)):
    if x != y: print('  diff', x, y)
