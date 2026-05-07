# license-auditor — Apache-2.0 vs deps

## Phase 1 — Tool
fossa / scancode / licensee **not installed**. Manual cross-reference.

## Phase 2 — license inventory

### Project license
- `LICENSE` (verified: Apache-2.0 v2.0, January 2004 header).
- `pyproject.toml:11` `license = "Apache-2.0"`. ✓
- Classifier `License :: OSI Approved :: Apache Software License` at
  `pyproject.toml:21`. ✓

### Dependency licenses (manual lookup against PyPI metadata)
| Package | License | Compatible with Apache-2.0? |
|---|---|---|
| pytest | MIT | yes |
| rich | MIT | yes |
| numpy | BSD-3-Clause | yes |
| torch (optional) | BSD-3-Clause | yes |
| cupy-cuda12x (optional) | MIT | yes |
| triton (optional) | MIT | yes |
| hypothesis (optional) | **MPL-2.0** | conditional — see below |
| ruff (dev) | MIT | yes (dev-only) |
| mypy (dev) | MIT | yes (dev-only) |
| pytest-cov (dev) | MIT | yes (dev-only) |

### MPL-2.0 (hypothesis) compatibility
MPL-2.0 is a *file-level* copyleft. Compatible with Apache-2.0 in
**aggregate** distribution (a wheel that imports hypothesis is fine).
The MPL-2.0 obligation only triggers if gpucheck *modifies hypothesis
source files*. gpucheck does not — hypothesis is imported as a library.
**Status**: COMPATIBLE, no action required.

### Forward-looking: `[mps]` / `[apple]` extras
Likely candidates and their licenses:
| Candidate | License | Status |
|---|---|---|
| pyobjc-core | MIT | yes |
| pyobjc-framework-Metal | MIT | yes |
| pyobjc-framework-MetalKit | MIT | yes |
| pyobjc-framework-MetalPerformanceShaders | MIT | yes |
| mlx | MIT | yes |
| coremltools | BSD-3-Clause | yes |

All MIT/BSD — Apache-2.0 compatible. **No GPL/AGPL/SSPL/ELv2 in the
realistic candidate set**. The implementer must still verify each
package's license **at the version pinned**.

### Red-flag scan
- AGPL-3.0: none.
- GPL-2.0 / GPL-3.0: none.
- SSPL: none.
- Elastic License 2.0: none.
- "No license" / "Custom": none.
- "Other/Proprietary": none.

## Phase 3 — verification

| Concern | Status |
|---|---|
| Project's own license file present | yes, Apache-2.0 |
| `pyproject.toml` license field matches LICENSE | yes |
| Any incompatible runtime dep | **no** |
| Any incompatible dev dep | no |
| Future MPS extras at risk of incompatibility | **no** (assuming Apple's pyobjc + mlx) |

## Output verdict
**PASS** — license posture is clean.

## SBOM summary (current `[project.dependencies]` only)
```
gpucheck@0.1.0  Apache-2.0
├── pytest      MIT          OSI: yes  compat: yes
├── rich        MIT          OSI: yes  compat: yes
└── numpy       BSD-3-Clause OSI: yes  compat: yes
```

## One advisory
- **LIC-1 (LOW)**: `pyproject.toml` does not declare a `license-files`
  field per PEP 639 (which entered packaging spec in 2024). Modern
  build tooling will warn that `LICENSE` is auto-included only by
  hatchling default. Add `license-files = ["LICENSE"]` for forward
  compatibility.
- **Location**: `pyproject.toml:11`
- **Remediation**:
```toml
license = "Apache-2.0"
license-files = ["LICENSE"]
```
