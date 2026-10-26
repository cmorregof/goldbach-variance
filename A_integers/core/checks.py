import numpy as np
X = 1 << 24; lim = X+1
isp = np.ones(lim, bool); isp[:2] = False
for p in range(2, int(lim**0.5)+1):
    if isp[p]: isp[p*p::p] = False
pr = np.nonzero(isp)[0]
# menor primo de Goldbach y records (OEIS A025018/A025019)
best = np.zeros(lim, np.int64)
todo = np.arange(6, lim, 2)
rem = todo.copy()
for p in pr[1:]:
    if rem.size == 0: break
    ok = isp[rem - p] if p < rem.min() else None
    m = (rem - p >= 2)
    hit = np.zeros(rem.size, bool); hit[m] = isp[rem[m]-p]
    best[rem[hit]] = p; rem = rem[~hit]
rec = []; mx = 0
for n in range(6, lim, 2):
    if best[n] > mx: mx = best[n]; rec.append((n, int(mx)))
print("records (2m, p) for 2m <= 2^24:", rec)
print("max p(2m) for 2m<=10^6:", best[6:10**6+1:2].max())
# Tabla 2.1
def G(n): return [p for p in pr[pr < n] if isp[n-p]]
def Gs(n): return [p for p in pr[pr <= n//2] if isp[n-p]]
for n in (32,38,100,556,1000,10000):
    print(n, len(Gs(n)), len(G(n)))
# cap 15: soporte de A = W R W con pesos sqrt(Lambda): pares de potencias de primo
def isprimepower(x):
    for p in pr[pr <= x]:
        if x % p == 0:
            while x % p == 0: x //= p
            return x == 1
    return False
bad = []
for n in range(6, 400, 2):
    pp = sum(1 for x in range(2, n-1) if isprimepower(x) and isprimepower(n-x))
    pp_primes = sum(1 for x in range(2, n-1) if isp[x] and isp[n-x])
    if pp != pp_primes: bad.append(n)
print("2m<400 where rank(A) != rank(B) (prime-power pairs exist):", bad[:15], len(bad))
