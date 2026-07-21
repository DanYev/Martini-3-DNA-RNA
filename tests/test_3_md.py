#!/usr/bin/env python3
"""
Integration tests: run minimal MD simulations for DNA and RNA to verify
the full martinize → solvate → EM → MD pipeline works end-to-end.
"""

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest
from smartini.utils import cd, gmx

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).parent
SCRIPT_DIR = TESTS_DIR.parent
DNA_SCRIPT = SCRIPT_DIR / "martinize_dna_alpha.py"
RNA_SCRIPT = SCRIPT_DIR / "martinize_rna_v3.0.0.py"
DNA_ITPS = SCRIPT_DIR / "dna_alpha_itps"
RNA_ITPS = SCRIPT_DIR / "rna_v3.0.0_itps"
CGMD_PARAMS = TESTS_DIR / "cgmd_params"
INPUT_DNA_PDB = TESTS_DIR / "dsDNA.pdb"
INPUT_RNA_PDB = TESTS_DIR / "dsRNA.pdb"

# Common Martini FF files in tests/cgmd_params
COMMON_FF = [
    "martini_v3.0.0.itp",
    "martini_v3.0.0_ions_v1.itp",
    "martini_v3.0.0_solvents_v1.itp",
]
# DNA/RNA-specific FF files
DNA_FF_FILE = "martini_alpha_dna.itp"
RNA_FF_FILE = "martini_v3.0.0_rna.itp"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _setup_system(itp_dir, nuc_ff_file, cg_pdb, cg_itp, molname, label, gmx_dir):
    """Set up solvated system: symlink FF ITPs, solvate, create system.top."""
    topdir = gmx_dir / "topol"
    topdir.mkdir()

    # Symlink common Martini FF files from tests/cgmd_params
    for ff_name in COMMON_FF:
        src = CGMD_PARAMS / ff_name
        dst = topdir / ff_name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    # Symlink DNA/RNA-specific FF file
    nuc_src = itp_dir / nuc_ff_file
    nuc_dst = topdir / nuc_ff_file
    if not nuc_dst.exists():
        nuc_dst.symlink_to(nuc_src.resolve())

    # Symlink molecule ITP
    mol_dst = topdir / f"{molname}.itp"
    if not mol_dst.exists():
        mol_dst.symlink_to(cg_itp.resolve())

    # Write system.top — nuc_ff before ions/solvents (they have [ moleculetype ])
    systop = gmx_dir / "system.top"
    with open(systop, "w", encoding="utf-8") as f:
        f.write('#include "topol/martini_v3.0.0.itp"\n')
        f.write(f'#include "topol/{nuc_ff_file}"\n')
        f.write('#include "topol/martini_v3.0.0_ions_v1.itp"\n')
        f.write('#include "topol/martini_v3.0.0_solvents_v1.itp"\n')
        f.write(f'\n#include "topol/{molname}.itp"\n\n')
        f.write("[ system ]\n")
        f.write(f"Martini {label} test system\n\n")
        f.write("[ molecules ]\n")
        f.write(f"{molname}    1\n")

    # Solvate
    solupdb = gmx_dir / "solute.pdb"
    shutil.copy(cg_pdb, solupdb)
    water_src = CGMD_PARAMS / "water.gro"
    shutil.copy(water_src, gmx_dir / "water.gro")

    with cd(gmx_dir):
        gmx("editconf", f=solupdb, o=solupdb, d="1.0", bt="cubic")
        gmx("solvate", cp=solupdb, cs="water.gro", p=systop,
            o="system.pdb", radius="0.17")
        gmx("editconf", f="system.pdb", o="system.gro")

    assert (gmx_dir / "system.gro").exists(), f"{label}: system.gro not created"


def _run_md(gmx_dir, label):
    """Run EM + 1000-step MD. Symlinks topol into rundir for include resolution."""
    mdpdir = gmx_dir / "mdp"
    mdpdir.mkdir()
    shutil.copy(CGMD_PARAMS / "mdp_em.mdp", mdpdir / "mdp_em.mdp")
    shutil.copy(CGMD_PARAMS / "mdp_md.mdp", mdpdir / "mdp_md.mdp")

    rundir = gmx_dir / "mdrun"
    rundir.mkdir()
    systop = gmx_dir / "system.top"
    topdir = gmx_dir / "topol"
    # symlink from rundir to gmx_dir/topol for include resolution
    (rundir / "topol").symlink_to(os.path.relpath(topdir, rundir), target_is_directory=True)

    with cd(rundir):
        gmx("grompp", f=mdpdir / "mdp_em.mdp", c=gmx_dir / "system.gro",
            r=gmx_dir / "system.gro", p=systop, o="em.tpr")
        gmx("mdrun", deffnm="em", ntomp=1)
    assert (rundir / "em.gro").exists(), f"{label}: EM failed"

    with cd(rundir):
        gmx("grompp", f=mdpdir / "mdp_md.mdp", c="em.gro", r="em.gro",
            p=systop, o="md.tpr", maxwarn="1")
        gmx("mdrun", deffnm="md", ntomp=1, nsteps=1000)
    assert (rundir / "md.gro").exists(), f"{label}: MD failed"


# ---------------------------------------------------------------------------
# Module fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dna_mod():
    return _load_module("martinize_dna_alpha", DNA_SCRIPT)


@pytest.fixture(scope="module")
def rna_mod():
    return _load_module("martinize_rna_cur", RNA_SCRIPT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMDDNA:
    """Full-pipeline MD test for DNA."""

    def test_full_pipeline(self, dna_mod, tmp_path):
        label = "dna_md"
        out_dir = tmp_path / label
        out_dir.mkdir()
        gmx_dir = out_dir / "gmx"
        gmx_dir.mkdir()
        cg_pdb = out_dir / "DNA.pdb"
        cg_itp = out_dir / "DNA.itp"

        dna_mod.martinize_dna(
            input_pdb=str(INPUT_DNA_PDB),
            output_structure=str(cg_pdb),
            output_topology=str(cg_itp),
            molecule_name="DNA",
        )
        assert cg_pdb.exists() and cg_itp.exists()
        _setup_system(DNA_ITPS, DNA_FF_FILE, cg_pdb, cg_itp, "DNA", label, gmx_dir)
        _run_md(gmx_dir, label)


class TestMDRNA:
    """Full-pipeline MD test for RNA."""

    def test_full_pipeline(self, rna_mod, tmp_path):
        label = "rna_md"
        out_dir = tmp_path / label
        out_dir.mkdir()
        gmx_dir = out_dir / "gmx"
        gmx_dir.mkdir()
        cg_pdb = out_dir / "RNA.pdb"
        cg_itp = out_dir / "RNA.itp"

        rna_mod.martinize_rna(
            input_pdb=str(INPUT_RNA_PDB),
            output_structure=str(cg_pdb),
            output_topology=str(cg_itp),
            molecule_name="RNA",
        )
        assert cg_pdb.exists() and cg_itp.exists()
        _setup_system(RNA_ITPS, RNA_FF_FILE, cg_pdb, cg_itp, "RNA", label, gmx_dir)
        _run_md(gmx_dir, label)
