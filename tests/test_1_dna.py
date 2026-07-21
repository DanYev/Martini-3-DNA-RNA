#!/usr/bin/env python3
"""
Smoke test for martinize_dna_alpha.py — verifies it generates PDB and ITP output.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
SCRIPT_DIR = TESTS_DIR.parent
DNA_SCRIPT = SCRIPT_DIR / "martinize_dna_alpha.py"
INPUT_PDB = TESTS_DIR / "dsDNA.pdb"


def _load_module(name, path):
    """Load a Python script as a module."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dna_mod():
    """Load the DNA martinize script."""
    return _load_module("martinize_dna_alpha", DNA_SCRIPT)


class TestSmokeDNA:
    """Verify the DNA script generates PDB and ITP output."""

    def test_generates_output(self, dna_mod, tmp_path):
        pdb_out = tmp_path / "dna_out.pdb"
        itp_out = tmp_path / "dna_out.itp"

        dna_mod.martinize_dna(
            input_pdb=str(INPUT_PDB),
            output_structure=str(pdb_out),
            output_topology=str(itp_out),
            molecule_name="DNA",
            elastic_network="no",
            debug=False,
        )

        assert pdb_out.exists(), "DNA PDB output not created"
        assert itp_out.exists(), "DNA ITP output not created"
        assert pdb_out.stat().st_size > 0, "DNA PDB output is empty"
        assert itp_out.stat().st_size > 0, "DNA ITP output is empty"

        pdb_text = pdb_out.read_text()
        itp_text = itp_out.read_text()
        assert "ATOM" in pdb_text, "DNA PDB has no ATOM lines"
        assert "[ atoms ]" in itp_text, "DNA ITP has no atoms section"
