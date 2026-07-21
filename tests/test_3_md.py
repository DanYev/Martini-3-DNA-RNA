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
INPUT_DNA_PDB = TESTS_DIR / "dsDNA.pdb"
INPUT_RNA_PDB = TESTS_DIR / "dsRNA.pdb"
CGMD_PARAMS = SCRIPT_DIR / "package" / "cgmd_params"


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


def _run_martinize_and_md(mod, input_pdb, molname, label, tmp_path):
    """Run martinize + minimal MD and return the run directory.

    Pipeline:
        1. martinize → CG PDB + ITP
        2. editconf + solvate
        3. EM (grompp + mdrun)
        4. Short MD (1000 steps)
    """
    out_dir = tmp_path / label
    out_dir.mkdir()
    gmx_dir = out_dir / "gmx"
    gmx_dir.mkdir()

    # --- Step 1: martinize ---
    cg_pdb = out_dir / f"{molname}.pdb"
    cg_itp = out_dir / f"{molname}.itp"

    # Determine which martinize function to call
    if hasattr(mod, "martinize_dna"):
        mod.martinize_dna(
            input_pdb=str(input_pdb),
            output_structure=str(cg_pdb),
            output_topology=str(cg_itp),
            molecule_name=molname,
            elastic_network="no",
            debug=False,
        )
    elif hasattr(mod, "martinize_rna"):
        mod.martinize_rna(
            input_pdb=str(input_pdb),
            output_structure=str(cg_pdb),
            output_topology=str(cg_itp),
            molecule_name=molname,
            elastic_network="no",
            debug=False,
        )

    assert cg_pdb.exists(), f"{label}: CG PDB not created"
    assert cg_itp.exists(), f"{label}: CG ITP not created"

    # --- Step 2: system setup ---
    topdir = gmx_dir / "topol"
    topdir.mkdir()

    # Copy force field ITPs
    ff_files = [
        "martini_v3.0.0.itp",
        "martini_alpha_nuc.itp",
        "martini_v3.0.0_ions_v1.itp",
        "martini_v3.0.0_solvents_v1.itp",
        "martini_v3.0.0_nucleobases_v1.itp",
    ]
    for ff_name in ff_files:
        ff_src = CGMD_PARAMS / ff_name
        if ff_src.exists():
            shutil.copy(ff_src, topdir / ff_name)

    # Copy molecule ITP
    shutil.copy(cg_itp, topdir / f"{molname}.itp")

    # Write system.top
    systop = gmx_dir / "system.top"
    with open(systop, "w", encoding="utf-8") as f:
        f.write('#include "martini_v3.0.0.itp"\n')
        f.write('#include "martini_alpha_nuc.itp"\n')
        f.write('#include "martini_v3.0.0_ions_v1.itp"\n')
        f.write('#include "martini_v3.0.0_solvents_v1.itp"\n')
        f.write('#include "martini_v3.0.0_nucleobases_v1.itp"\n')
        f.write(f'\n#include "topol/{molname}.itp"\n\n')
        f.write("[ system ]\n")
        f.write(f"Martini {label} test system\n\n")
        f.write("[ molecules ]\n")
        f.write(f"{molname}    1\n")

    # Solvate — use absolute path for water.gro
    solupdb = gmx_dir / "solute.pdb"
    shutil.copy(cg_pdb, solupdb)
    water_path = str((CGMD_PARAMS / "water.gro").resolve())

    with cd(gmx_dir):
        gmx("editconf", f=solupdb, o=solupdb, d="1.0", bt="cubic")
        gmx("solvate", cp=solupdb, cs=water_path, p=systop,
            o="system.pdb", radius="0.17")
        gmx("editconf", f="system.pdb", o="system.gro")

    assert (gmx_dir / "system.gro").exists(), f"{label}: system.gro not created"

    # --- Step 3: Energy Minimization ---
    mdpdir = gmx_dir / "mdp"
    mdpdir.mkdir()
    shutil.copy(CGMD_PARAMS / "mdp_em.mdp", mdpdir / "mdp_em.mdp")

    rundir = gmx_dir / "mdrun"
    rundir.mkdir()

    with cd(rundir):
        gmx("grompp", f=mdpdir / "mdp_em.mdp", c=gmx_dir / "system.gro",
            r=gmx_dir / "system.gro", p=systop, o="em.tpr")
        gmx("mdrun", deffnm="em", ntomp=1)

    assert (rundir / "em.gro").exists(), f"{label}: EM output not created"

    # --- Step 4: MD (1000 steps) ---
    shutil.copy(CGMD_PARAMS / "mdp_md.mdp", mdpdir / "mdp_md.mdp")

    with cd(rundir):
        gmx("grompp", f=mdpdir / "mdp_md.mdp", c="em.gro", r="em.gro",
            p=systop, o="md.tpr", maxwarn="1")
        gmx("mdrun", deffnm="md", ntomp=1, nsteps=1000)

    assert (rundir / "md.gro").exists(), f"{label}: MD output not created"

    return gmx_dir


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
        _run_martinize_and_md(dna_mod, INPUT_DNA_PDB, "DNA", "dna_md", tmp_path)


class TestMDRNA:
    """Full-pipeline MD test for RNA."""

    def test_full_pipeline(self, rna_mod, tmp_path):
        _run_martinize_and_md(rna_mod, INPUT_RNA_PDB, "RNA", "rna_md", tmp_path)
