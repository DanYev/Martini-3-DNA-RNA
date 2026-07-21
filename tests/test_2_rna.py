#!/usr/bin/env python3
"""
Unit tests for martinize_rna_v3.0.0.py

Compares the ref (hardcoded) script output with the current (file-based) script
output to ensure the migration to map/ITP files produces identical results.

Run with: pytest test_rna.py -v
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).parent
SCRIPT_DIR = TESTS_DIR.parent
REF_SCRIPT = TESTS_DIR / "martinize_rna_ref.py"
CUR_SCRIPT = SCRIPT_DIR / "martinize_rna_v3.0.0.py"
INPUT_PDB = TESTS_DIR / "dsRNA.pdb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_module(name, path):
    """Load a Python script as a module."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_pdb_atoms(pdb_path):
    """Parse ATOM/HETATM lines from a PDB file into a list of dicts."""
    atoms = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atoms.append({
                    "atid": int(line[6:11].strip()),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chid": line[21:22].strip(),
                    "resid": int(line[22:26].strip()),
                    "x": float(line[30:38].strip()),
                    "y": float(line[38:46].strip()),
                    "z": float(line[46:54].strip()),
                })
    return atoms


def parse_itp(itp_path):
    """Parse an ITP file into a dict of sections with raw line lists."""
    sections = {}
    header = []
    current_section = "header"

    with open(itp_path) as f:
        for raw in f:
            stripped = raw.strip()
            if stripped.startswith(";") or stripped.startswith("#"):
                if current_section == "header":
                    header.append(raw.rstrip("\n"))
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1].strip()
                if current_section not in sections:
                    sections[current_section] = []
                continue
            if current_section not in sections:
                sections[current_section] = []
            if stripped == "":
                continue
            sections[current_section].append(raw.rstrip("\n"))

    sections["_header"] = header
    return sections


def parse_itp_atoms(itp_path):
    """Parse atoms section of ITP into structured list."""
    itp = parse_itp(itp_path)
    atom_lines = itp.get("atoms", [])
    atoms = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) >= 8:
            atoms.append({
                "id": int(parts[0]),
                "type": parts[1],
                "resid": int(parts[2]),
                "resname": parts[3],
                "name": parts[4],
                "cgnr": int(parts[5]),
                "charge": float(parts[6]),
                "mass": float(parts[7]) if len(parts) > 7 else None,
            })
    return atoms


def parse_bonded_section(itp_path, section_name):
    """Parse a bonded section into structured list of numeric tuples."""
    itp = parse_itp(itp_path)
    lines = itp.get(section_name, [])
    entries = []
    for line in lines:
        if line.strip().startswith("#"):
            continue
        data = line.split(";")[0].strip()
        parts = data.split()
        if not parts:
            continue
        nums = [float(p) if "." in p or "e" in p.lower() else int(p) for p in parts]
        entries.append(tuple(nums))
    return entries


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ref():
    """Load the reference (hardcoded) martinize RNA script."""
    return _load_module("martinize_rna_ref", REF_SCRIPT)


@pytest.fixture(scope="module")
def cur():
    """Load the current (file-based) martinize RNA script."""
    return _load_module("martinize_rna_cur", CUR_SCRIPT)


@pytest.fixture(scope="module")
def ref_output(ref, tmp_path_factory):
    """Run ref script on dsRNA.pdb, return (pdb_path, itp_path)."""
    out_dir = tmp_path_factory.mktemp("ref")
    pdb_out = out_dir / "ref.pdb"
    itp_out = out_dir / "ref.itp"

    old_cwd = os.getcwd()
    os.chdir(str(TESTS_DIR))
    try:
        ref.martinize_rna(
            input_pdb=str(INPUT_PDB),
            output_structure=str(pdb_out),
            output_topology=str(itp_out),
            molecule_name="RNA",
            elastic_network="no",
            debug=False,
        )
    finally:
        os.chdir(old_cwd)
    return str(pdb_out), str(itp_out)


@pytest.fixture(scope="module")
def cur_output(cur, tmp_path_factory):
    """Run current script on dsRNA.pdb, return (pdb_path, itp_path)."""
    out_dir = tmp_path_factory.mktemp("cur")
    pdb_out = out_dir / "cur.pdb"
    itp_out = out_dir / "cur.itp"

    itp_dir = str(SCRIPT_DIR / "rna_v3.0.0_itps")

    old_cwd = os.getcwd()
    os.chdir(str(SCRIPT_DIR))
    try:
        cur.martinize_rna(
            input_pdb=str(INPUT_PDB),
            output_structure=str(pdb_out),
            output_topology=str(itp_out),
            molecule_name="RNA",
            elastic_network="no",
            itp_dir=itp_dir,
            debug=False,
        )
    finally:
        os.chdir(old_cwd)
    return str(pdb_out), str(itp_out)


# ---------------------------------------------------------------------------
# PDB comparison tests
# ---------------------------------------------------------------------------
class TestPDBComparison:
    """Compare PDB output between ref and current script."""

    def test_atom_count(self, ref_output, cur_output):
        ref_pdb, _ = ref_output
        cur_pdb, _ = cur_output
        ref_atoms = parse_pdb_atoms(ref_pdb)
        cur_atoms = parse_pdb_atoms(cur_pdb)
        assert len(ref_atoms) == len(cur_atoms), \
            f"Atom count mismatch: ref={len(ref_atoms)}, cur={len(cur_atoms)}"

    def test_atom_identities(self, ref_output, cur_output):
        ref_pdb, _ = ref_output
        cur_pdb, _ = cur_output
        ref_atoms = parse_pdb_atoms(ref_pdb)
        cur_atoms = parse_pdb_atoms(cur_pdb)

        for i, (ra, ca) in enumerate(zip(ref_atoms, cur_atoms)):
            assert ra["atid"] == ca["atid"], \
                f"Atom {i}: ID mismatch ref={ra['atid']} cur={ca['atid']}"
            assert ra["name"] == ca["name"], \
                f"Atom {i}: name mismatch ref='{ra['name']}' cur='{ca['name']}'"
            assert ra["resname"] == ca["resname"], \
                f"Atom {i}: resname mismatch ref='{ra['resname']}' cur='{ca['resname']}'"
            assert ra["resid"] == ca["resid"], \
                f"Atom {i}: resid mismatch ref={ra['resid']} cur={ca['resid']}"

    def test_atom_coordinates(self, ref_output, cur_output):
        ref_pdb, _ = ref_output
        cur_pdb, _ = cur_output
        ref_atoms = parse_pdb_atoms(ref_pdb)
        cur_atoms = parse_pdb_atoms(cur_pdb)

        for i, (ra, ca) in enumerate(zip(ref_atoms, cur_atoms)):
            assert ra["x"] == pytest.approx(ca["x"], abs=1e-4), \
                f"Atom {i} ({ra['name']}): X mismatch ref={ra['x']} cur={ca['x']}"
            assert ra["y"] == pytest.approx(ca["y"], abs=1e-4), \
                f"Atom {i} ({ra['name']}): Y mismatch ref={ra['y']} cur={ca['y']}"
            assert ra["z"] == pytest.approx(ca["z"], abs=1e-4), \
                f"Atom {i} ({ra['name']}): Z mismatch ref={ra['z']} cur={ca['z']}"


# ---------------------------------------------------------------------------
# ITP comparison tests
# ---------------------------------------------------------------------------
class TestITPComparison:
    """Compare ITP output section by section."""

    def test_header_lines(self, ref_output, cur_output):
        _, ref_itp = ref_output
        _, cur_itp = cur_output
        ref_sections = parse_itp(ref_itp)
        cur_sections = parse_itp(cur_itp)

        assert len(ref_sections["_header"]) > 0, "Ref has no header lines"
        assert len(cur_sections["_header"]) > 0, "Cur has no header lines"

        ref_header_text = "\n".join(ref_sections["_header"])
        cur_header_text = "\n".join(cur_sections["_header"])
        assert "MARTINI" in ref_header_text
        assert "MARTINI" in cur_header_text
        assert "Sequence:" in ref_header_text
        assert "Sequence:" in cur_header_text

    def test_moleculetype(self, ref_output, cur_output):
        _, ref_itp = ref_output
        _, cur_itp = cur_output
        ref_sec = parse_itp(ref_itp)
        cur_sec = parse_itp(cur_itp)
        assert ref_sec.get("moleculetype", []) == cur_sec.get("moleculetype", [])

    def test_atoms_section(self, ref_output, cur_output):
        _, ref_itp = ref_output
        _, cur_itp = cur_output
        ref_atoms = parse_itp_atoms(ref_itp)
        cur_atoms = parse_itp_atoms(cur_itp)
        assert len(ref_atoms) == len(cur_atoms)
        for i, (ra, ca) in enumerate(zip(ref_atoms, cur_atoms)):
            assert ra == ca, f"Atom {i} differs:\nref: {ra}\ncur: {ca}"

    @pytest.mark.parametrize("section", [
        "bonds", "angles", "dihedrals", "constraints",
        "exclusions", "pairs", "virtual_sites3",
    ])
    def test_bonded_section(self, ref_output, cur_output, section):
        _, ref_itp = ref_output
        _, cur_itp = cur_output
        ref_entries = sorted(parse_bonded_section(ref_itp, section))
        cur_entries = sorted(parse_bonded_section(cur_itp, section))
        assert len(ref_entries) == len(cur_entries), \
            f"[{section}] count mismatch: ref={len(ref_entries)} cur={len(cur_entries)}"
        for i, (re, ce) in enumerate(zip(ref_entries, cur_entries)):
            assert re == ce, \
                f"[{section}] entry {i} differs:\nref: {re}\ncur: {ce}"

    def test_same_sections_present(self, ref_output, cur_output):
        _, ref_itp = ref_output
        _, cur_itp = cur_output
        ref_sec = parse_itp(ref_itp)
        cur_sec = parse_itp(cur_itp)
        ref_keys = set(ref_sec.keys()) - {"_header"}
        cur_keys = set(cur_sec.keys()) - {"_header"}
        assert ref_keys == cur_keys, \
            f"Section mismatch:\nref only: {ref_keys - cur_keys}\ncur only: {cur_keys - ref_keys}"

    def test_line_by_line_identical(self, ref_output, cur_output):
        """Read both ITP files line by line and compare.

        Only the timestamp and arguments header lines are allowed to differ.
        """
        _, ref_itp = ref_output
        _, cur_itp = cur_output

        with open(ref_itp) as rf, open(cur_itp) as cf:
            ref_lines = rf.readlines()
            cur_lines = cf.readlines()

        assert len(ref_lines) == len(cur_lines), (
            f"Line count mismatch: ref={len(ref_lines)} cur={len(cur_lines)}"
        )

        mismatches = []
        for i, (rl, cl) in enumerate(zip(ref_lines, cur_lines), start=1):
            if rl == cl:
                continue
            rls = rl.strip()
            cls = cl.strip()
            if rls.startswith("; Generated on:") and cls.startswith("; Generated on:"):
                continue
            if rls.startswith("; input_pdb=") and cls.startswith("; input_pdb="):
                continue
            mismatches.append((i, rl.rstrip("\n"), cl.rstrip("\n")))

        if mismatches:
            msg_parts = [f"{len(mismatches)} line(s) differ:"]
            for lineno, ref_line, cur_line in mismatches[:20]:
                msg_parts.append(f"  Line {lineno}:")
                msg_parts.append(f"    ref: {ref_line}")
                msg_parts.append(f"    cur: {cur_line}")
            if len(mismatches) > 20:
                msg_parts.append(f"  ... and {len(mismatches) - 20} more")
            pytest.fail("\n".join(msg_parts))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
class TestSmokeRNA:
    """Verify the RNA script generates PDB and ITP output."""

    def test_generates_output(self, cur, tmp_path):
        pdb_out = tmp_path / "rna_out.pdb"
        itp_out = tmp_path / "rna_out.itp"

        cur.martinize_rna(
            input_pdb=str(INPUT_PDB),
            output_structure=str(pdb_out),
            output_topology=str(itp_out),
            molecule_name="RNA",
            elastic_network="no",
            debug=False,
        )

        assert pdb_out.exists(), "RNA PDB output not created"
        assert itp_out.exists(), "RNA ITP output not created"
        assert pdb_out.stat().st_size > 0, "RNA PDB output is empty"
        assert itp_out.stat().st_size > 0, "RNA ITP output is empty"

        pdb_text = pdb_out.read_text()
        itp_text = itp_out.read_text()
        assert "ATOM" in pdb_text, "RNA PDB has no ATOM lines"
        assert "[ atoms ]" in itp_text, "RNA ITP has no atoms section"
