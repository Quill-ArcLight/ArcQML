#!/usr/bin/env python3
"""Generate one ArcQML-compatible JSON Hamiltonian per STO-3G molecule."""

from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pennylane
from scipy import sparse
from scipy.sparse.linalg import eigsh

BOHR_PER_ANGSTROM = 1.8897261254578281
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"

MOLECULES = {
    "h2": {
        "label": "H2",
        "symbols": ["H", "H"],
        "coordinates_angstrom": [[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]],
        "active_electrons": 2,
        "active_orbitals": 2,
    },
}


def pauli_terms(operator: pennylane.operation.Operator) -> list[dict]:
    sentence = pennylane.pauli.pauli_sentence(operator)
    terms = []
    for word, coefficient in sentence.items():
        factors = sorted(word.items())
        terms.append(
            {
                "coefficient": float(np.real(coefficient)),
                "paulis": "".join(symbol for _, symbol in factors),
                "qubits": [int(wire) for wire, _ in factors],
            }
        )
    terms.sort(key=lambda term: (len(term["qubits"]), term["qubits"], term["paulis"]))
    return terms


def fixed_particle_basis(num_qubits: int, electrons: int) -> list[int]:
    return [
        sum(1 << qubit for qubit in occupied)
        for occupied in itertools.combinations(range(num_qubits), electrons)
    ]


def apply_pauli_word(state: int, item: dict) -> tuple[int, complex]:
    target = state
    phase = 1.0 + 0.0j
    for symbol, qubit in zip(item["paulis"], item["qubits"]):
        occupied = bool(state & (1 << qubit))
        if symbol == "X":
            target ^= 1 << qubit
        elif symbol == "Y":
            target ^= 1 << qubit
            phase *= -1j if occupied else 1j
        elif symbol == "Z" and occupied:
            phase *= -1.0
    return target, phase


def fixed_particle_matrix(
    num_qubits: int,
    electrons: int,
    terms: list[dict],
) -> sparse.csr_matrix:
    basis = fixed_particle_basis(num_qubits, electrons)
    indices = {state: index for index, state in enumerate(basis)}
    rows, columns, values = [], [], []
    for column, state in enumerate(basis):
        for item in terms:
            target, phase = apply_pauli_word(state, item)
            row = indices.get(target)
            if row is not None:
                rows.append(row)
                columns.append(column)
                values.append(item["coefficient"] * phase)
    dimension = len(basis)
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()


def full_fock_matrix(num_qubits: int, terms: list[dict]) -> sparse.csr_matrix:
    dimension = 1 << num_qubits
    columns = np.arange(dimension, dtype=np.int64)
    parity = np.asarray([index.bit_count() & 1 for index in range(dimension)])
    grouped: dict[int, list[tuple[float, int, int]]] = {}
    for item in terms:
        x_mask = 0
        z_mask = 0
        y_count = 0
        for symbol, qubit in zip(item["paulis"], item["qubits"]):
            if symbol in "XY":
                x_mask |= 1 << qubit
            if symbol in "YZ":
                z_mask |= 1 << qubit
            if symbol == "Y":
                y_count += 1
        grouped.setdefault(x_mask, []).append(
            (item["coefficient"], z_mask, y_count)
        )

    row_blocks, column_blocks, value_blocks = [], [], []
    for x_mask, group in grouped.items():
        values = np.zeros(dimension, dtype=np.complex128)
        for coefficient, z_mask, y_count in group:
            signs = 1.0 - 2.0 * parity[np.bitwise_and(columns, z_mask)]
            values += coefficient * (1j**y_count) * signs
        nonzero = np.abs(values) > 1.0e-14
        row_blocks.append(np.bitwise_xor(columns[nonzero], x_mask))
        column_blocks.append(columns[nonzero])
        value_blocks.append(values[nonzero])

    return sparse.coo_matrix(
        (
            np.concatenate(value_blocks),
            (np.concatenate(row_blocks), np.concatenate(column_blocks)),
        ),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()


def diagonal_energy(state: int, terms: list[dict]) -> float:
    value = 0.0
    for item in terms:
        target, phase = apply_pauli_word(state, item)
        if target == state:
            value += item["coefficient"] * phase.real
    return value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, specification in MOLECULES.items():
        coordinates = np.asarray(specification["coordinates_angstrom"]) * BOHR_PER_ANGSTROM
        molecule = pennylane.qchem.Molecule(
            specification["symbols"], coordinates, basis_name="sto-3g"
        )
        hamiltonian, num_qubits = pennylane.qchem.molecular_hamiltonian(
            molecule,
            method="dhf",
            active_electrons=specification["active_electrons"],
            active_orbitals=specification["active_orbitals"],
            mapping="jordan_wigner",
        )
        terms = pauli_terms(hamiltonian)
        matrix = fixed_particle_matrix(
            num_qubits,
            specification["active_electrons"],
            terms,
        )
        occupied_index = sum(1 << qubit for qubit in range(specification["active_electrons"]))
        if matrix.shape[0] <= 512:
            exact_energy = float(np.linalg.eigvalsh(matrix.toarray())[0])
        else:
            exact_energy = float(eigsh(matrix, k=1, which="SA", return_eigenvectors=False)[0])
        full_matrix = full_fock_matrix(num_qubits, terms)
        full_ground_energy = float(
            eigsh(full_matrix, k=1, which="SA", return_eigenvectors=False)[0]
        )
        # Keep the calculation format above, but export the public ArcQML schema.
        output = {
            "num_qubits": num_qubits,
            "terms": [
                {
                    "coefficient": item["coefficient"],
                    "paulis": [
                        {"qubit": qubit, "pauli": symbol}
                        for qubit, symbol in zip(item["qubits"], item["paulis"])
                    ],
                }
                for item in terms
            ],
        }
        print(
            f"{specification['label']}: qubits={num_qubits}, terms={len(terms)}, "
            f"E0({specification['active_electrons']}e)={exact_energy:.12f} Ha, "
            f"E0(full)={full_ground_energy:.12f} Ha, "
            f"E(HF)={diagonal_energy(occupied_index, terms):.12f} Ha"
        )
        output_path = OUTPUT_DIR / f"{key}_hamiltonian.json"
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
