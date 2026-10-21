# pandora-visibility
<a href="https://github.com/pandoramission/pandora-visibility/actions/workflows/python-package.yml"><img src="https://github.com/pandoramission/pandora-visibility/actions/workflows/python-package.yml/badge.svg" alt="Test status"/></a>

Tool for calculating the visibility of targets from Pandora

See the docs/ folder for an example notebook on using this software.

### NIRDA trace check and the offline Gaia catalog
By default `Visibility.get_visibility(..., optimize_roll=True)` also rejects rolls where the NIRDA slitless traces of nearby stars land on the target's extraction aperture (`use_trace_sweep=True`, see `src/pandoravisibility/trace.py`). The nearby stars come from a local copy of Gaia DR3 managed by [gaiaoffline](https://github.com/christinahedges/gaiaoffline), which is installed with this package but whose database (~30 GB) must be downloaded once, as described in its README (a precomputed copy is on Zenodo). Without the database, construct `Visibility` with `use_trace_sweep=False`; the roll search then behaves as before.

The check takes `trace_contamination_tol` (default 1e-3, contaminating over target flux in the worst spectral channel), `nirda_roi_x` and `nirda_roi_y` (default 80 x 256). Every other model setting (roll-to-detector mapping, trace extent, aperture, line-spread function, roll uncertainty, catalog depth) is a module constant at the top of `trace.py` or `star_field.py`. `trace.plot_trace` draws the simulated subarray at a given roll.

### Acknowledgements
This project incorporates code from [m4opt](https://github.com/m4opt/m4opt/), which is licensed under the BSD 3-Clause License.
