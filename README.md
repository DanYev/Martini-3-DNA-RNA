## Martinize RNA/DNA Scripts

A standalone script that converts all-atom RNA structures to coarse-grained representations using the Martini 3 force field.

### Usage
1. Download the script and required `.itp` files from the `scripts/` directory
2. View available options:
```bash
python martinize_rna_v3.0.0.py --help
```

### Force Field Files

The script automatically loads bonded/unbonded parameters from:
```
/path/to/the/script/martinize_rna_v3.0.0_itps/
```
```

### Citation

If you use this script in your research, please cite:

> Yangaliev D, Ozkan SB. Coarse-grained RNA model for the Martini 3 force field. *Biophys J*. 2025 Aug 5:S0006-3495(25)00483-7. doi: [10.1016/j.bpj.2025.07.034](https://doi.org/10.1016/j.bpj.2025.07.034)


### DNA Script

The current version works (to out knowdledge) but is still in development. 
Feel free to contact us or report any issues.

### License

Copyright (c) 2025, DY
Licensed under the GNU General Public License v3.0

[![DOI](https://zenodo.org/badge/944535799.svg)](https://doi.org/10.5281/zenodo.19207978)

