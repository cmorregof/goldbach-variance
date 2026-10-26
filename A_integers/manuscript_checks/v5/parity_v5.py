# EN/ES parity check for v5 (copy of v4_scratch/parity_v4.py) (extends opus2_scratch/enes.py).
import re, collections, sys, json
base = '/home/claude/pruebas/manuscrito/'
en = open(base+'manuscrito_v5_en.tex').read(); es = open(base+'manuscrito_v5_es.tex').read()
strip = lambda s: re.sub(r'(?<!\\)%.*', '', s)
en, es = strip(en), strip(es)
body = lambda s: s[s.index(r'\begin{document}'):]
en, es = body(en), body(es)
out = {}
lab = lambda s: re.findall(r'\\label\{([^}]*)\}', s)
out['labels_equal_order'] = lab(en) == lab(es); out['n_labels'] = (len(lab(en)), len(lab(es)))
for kind in ['cite', 'ref', 'eqref']:
    f = lambda s: collections.Counter(x.strip() for m in re.findall(r'\\%s(?:\[[^\]]*\])?\{([^}]*)\}' % kind, s) for x in m.split(','))
    a, b = f(en), f(es)
    out[kind+'_equal'] = a == b
    if a != b: out[kind+'_diff'] = (dict(a-b), dict(b-a))
envs = lambda s: collections.Counter(re.findall(r'\\begin\{(theorem|proposition|lemma|corollary|claim|conjecture|hypothesis|definition|remark|proof|table)\}', s))
out['envs_equal'] = envs(en) == envs(es); out['envs'] = dict(envs(en))
def nums(s): return collections.Counter(re.findall(r'(?<![\w.])-?\d+\.\d+', s.replace('{,}', '.')))
a, b = nums(en), nums(es); out['decimals_equal'] = a == b
if a != b: out['decimals_diff'] = (dict(a-b), dict(b-a))
ints = lambda s: collections.Counter(re.findall(r'(?<![\w.{,])\d{2,}(?![\w.])', s.replace('{,}', '.')))
a, b = ints(en), ints(es); out['ints_equal'] = a == b
if a != b: out['ints_diff'] = (dict(a-b), dict(b-a))
st = lambda s: re.findall(r'\\st\{', s); out['status_lines'] = (len(st(en)), len(st(es)))
vf = lambda s: len(re.findall(r'\\verif\{\}', s)); out['verif_marks'] = (vf(en), vf(es))
bib = lambda s: re.findall(r'\\bibitem\{([^}]*)\}', s); out['bibitems_equal_order'] = bib(en) == bib(es); out['n_bibitems'] = len(bib(en))
seq = lambda s: re.findall(r'\\begin\{(theorem|proposition|lemma|corollary|claim|conjecture|hypothesis|definition|remark)\}(?:\[[^\]]*\])?\s*(?:\\label\{([^}]*)\})?', s)
out['env_label_sequence_equal'] = seq(en) == seq(es)
# cited keys all defined
cited = set(x.strip() for m in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]*)\}', en) for x in m.split(','))
out['cited_not_in_bib'] = sorted(cited - set(bib(en))); out['bib_not_cited'] = sorted(set(bib(en)) - cited)
# math segments: inline $...$ multiset
def maths(s):
    s = s.replace('{,}', '.')
    return collections.Counter(m.strip() for m in re.findall(r'(?<!\\)\$([^$]+)\$', s))
a, b = maths(en), maths(es)
out['inline_math_segments'] = (sum(a.values()), sum(b.values()))
d1, d2 = a-b, b-a
out['inline_math_EN_minus_ES'] = list(d1.elements()); out['inline_math_ES_minus_EN'] = list(d2.elements())
# display math blocks
def disp(s):
    s = s.replace('{,}', '.')
    blocks = re.findall(r'\\\[(.*?)\\\]', s, re.S) + re.findall(r'\\begin\{(?:align|align\*|gather|gather\*|multline|multline\*|equation)\}(.*?)\\end\{', s, re.S)
    return collections.Counter(re.sub(r'\s+', '', x) for x in blocks)
a, b = disp(en), disp(es); out['display_blocks_equal'] = a == b
if a != b: out['display_diff_counts'] = (sum((a-b).values()), sum((b-a).values()))
json.dump(out, open(base+'v5_scratch/parity_v5.json', 'w'), indent=1, ensure_ascii=False)
for k, v in out.items(): print(k, ':', v)
