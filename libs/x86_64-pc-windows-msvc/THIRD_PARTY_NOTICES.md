# Third-party materials

ArcQML's original framework source and official Runtime have separate licenses
in LICENSE and LICENSE-RUNTIME. Third-party rights are not relicensed by them.

Binary candidates include THIRD_PARTY_LICENSES.txt with versioned upstream
copyright, license, NOTICE and AUTHORS texts; Rust toolchain copyright and
license materials; and a CycloneDX SBOM. The SBOM states its scope, including
locked build/optional dependencies that need not all be linked into the binary.
NumPy is an external Python dependency installed separately under its own
license; it is not bundled in the ArcQML wheel. Consult its distribution notices.

Generated documentation in docs/doc includes third-party fonts and web assets.
Retain the accompanying OFL, MIT and COPYRIGHT notices with those assets.

The German Credit CSV in examples/data is the Open Data LMU dataset
"Kreditscoring zur Klassifikation von Kreditnehmern" (2010),
https://doi.org/10.5282/ubm/data.23, supplied under PDDL-1.0. All 1,000 rows and
21 numeric columns match the LMU download in their original order; the CSV
uses English headers and comma separators. See examples/data/README.md and
examples/data/LICENSE-PDDL. ArcQML's noncommercial source license does not
restrict the independent rights granted for this third-party dataset.

The H2 Hamiltonian was reproduced on 2026-09-15 from the supplied project
generator using PennyLane 0.45.1 (dhf), NumPy 2.4.6 and SciPy 1.17.1. All 15
Pauli coefficients match exactly. See examples/data/h2-generation-record.json
and examples/python/requirements-h2.txt. These are numerical calculation
outputs; third-party generator dependencies retain their own licenses.
