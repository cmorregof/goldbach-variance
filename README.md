# goldbach-variance

Code and data accompanying three papers by J. C. Riaño-Rojas and C. M. Orrego-Franco (Universidad Nacional de Colombia, Sede Manizales), in preparation, 2026:

- **A** — the mean square of the binary Goldbach error in the box: exact decomposition, endpoint obstruction and the large-divisor tail;
- **B** — the variance of Goldbach representations in F_q[T] in the large-q limit;
- **C** — the four-point singular series and the constant one half.

## Layout

| Folder | Contents |
|---|---|
| `A_integers/` | Exact Parseval decomposition and singular-series constants (`core/`), dyadic-block computations up to 2^28 and the Granville normalization (`core/`), moduli-range checks (`round3_moduli/`), large-divisor tail (`large_divisor_tail_E1/`), zero dictionary (`zero_dictionary_K1/`), and the scripts used to check numbers in successive manuscript versions (`manuscript_checks/`). |
| `B_function_fields/` | Exact computation of the four terms of the decomposition in F_q[T] (`audit/`), deterministic bookkeeping of the cross term (`gate/`), checks of the proof (`proof_checks/`), and the tables of paper B (`tables/`). |
| `C_four_point/` | The four-point singular-series model (`pretest/`), the Ramanujan-averaged part (`singular_series_E1/`), and the shift term (`lemma_G_run7/`, `lemma_G_run8/`). |
| `verification/` | Independent re-implementations used to check the new proofs and numbers. |

Each folder contains the Python scripts together with their JSON/text outputs, exactly as produced. Some scripts contain absolute paths from the environment in which they were run; adjust them before re-running. Several computations at the largest sizes (N up to 2^28) need tens of GB of memory or block-wise runs; the block-wise versions are included.

## Requirements

Python 3.10+ with NumPy, SciPy, mpmath and SymPy (see `requirements.txt`).

## Licence

Code: MIT (see `LICENSE`). Data and outputs: CC BY 4.0.

## How these computations were produced

See the AI Declaration in each paper.
