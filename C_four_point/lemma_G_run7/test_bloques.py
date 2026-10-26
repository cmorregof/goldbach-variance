"""[M] Control: prod de bloques de tau * rho = r (de e1_comun) para varios n."""
import numpy as np
from g_comun import r_and_rho, log_tau_blocks, primes_upto
pr = primes_upto(1 << 24)
for n, H in ((1000, 999), (30030, 5000), (65536, 20000), (999958, 3000)):
    r, rho = r_and_rho(n, H, pr)
    b = [5, 50, 1000, H + 1, 2 * n + 3, 1 << 40]
    lt = log_tau_blocks(n, H, b, pr)
    tau = np.exp(sum(lt))
    m = rho > 0
    print(n, H, "max rel err r vs rho*tau:", float(np.max(np.abs(rho[m] * tau[m] - r[m]) / r[m])), " zeros match:", bool(np.all((r == 0) == (rho == 0))))
