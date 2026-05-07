# crypto-reviewer — gpucheck v1.0

## Phase 1 — Tool
No dedicated tool. Grep-based scan for known-weak primitives.

```
$ grep -rn 'md5\|MD5\|sha1\|SHA1\|DES\|RC4\|RC2' src/
(no matches)

$ grep -rn 'hashlib\|hmac\|cryptography\|secrets\.\|random\.' src/
src/gpucheck/fuzzing/shapes.py:172:    rng = random.Random(seed)

$ grep -rn 'Math\.random\|rand()' src/
(no matches)
```

## Phase 2 — review

gpucheck has **no cryptographic surface**. It is a pytest plugin for
GPU correctness/perf testing. There are:
- no passwords, no auth tokens, no sessions, no JWTs;
- no encryption at rest of any data the library produces;
- no TLS / HTTPS anywhere (no network calls);
- no key management;
- no PII handling (other than user-controlled test names possibly
  echoed to console / JSON).

### Single PRNG site
`src/gpucheck/fuzzing/shapes.py:172` uses Python's
`random.Random(seed)` for fuzz-shape generation. This is a Mersenne
Twister, **not cryptographically secure**, but the use case is
*property-testing* — reproducible across runs. `random.Random` is
the right tool. Switching to `secrets` would defeat the purpose.

#### Verdict: NOT a finding.
- **Justification**: PRNG output influences only generated tensor
  shapes. There is no security decision based on the output. The
  `seed` is user-supplied or defaults to `None` (system-seeded).
- **Reference**: `random` docs explicitly recommend `secrets` only
  for cryptographic use.

## Phase 3 — verification

| Concern | Code site | Verdict |
|---|---|---|
| Weak hash | none | n/a |
| Weak symmetric cipher | none | n/a |
| Weak asymmetric / RSA <2048 | none | n/a |
| Hard-coded keys | none (verified, see secrets-hunter.md) | n/a |
| Insecure RNG used in security-relevant decision | none — PRNG is for shape fuzzing only | n/a |
| Plaintext PII storage | none | n/a |
| TLS misconfig | n/a, no TLS | n/a |
| Cookie flags | n/a, no HTTP | n/a |

## Output verdict
**PASS** — no crypto findings. The only PRNG use is correct.

## Forward-looking note for MPS work
If the MPS sanitizer ever computes a *content hash* of a buffer to
detect mutation between dispatches (an obvious choice for the N5
TOCTOU mitigation in threat-modeler.md), it should use **BLAKE3 or
SHA-256 from `hashlib`** — not MD5 — even though the use case is
"detect accidental change in tests", not adversarial. Using SHA-256
costs nothing extra on Apple Silicon (hardware AES + SHA), and avoids
future grep-noise from security tooling.
