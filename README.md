# csm_functions

A personal Python package of convenience functions for analyzing and plotting electrochemical and materials-characterization data. I built it over the course of my PhD at the Colorado School of Mines to streamline the analysis and figure-making I performed repeatedly, so that each new dataset could be processed and plotted consistently rather than from scratch.

## Structure

- `eis/`: functions for analyzing and plotting electrochemical impedance spectroscopy (EIS) data.
- `eis_hyb/`: Used to analyze EIS data and interfaces with the Hybrid-DRT package developed by Dr. Jake Huang (https://github.com/jdhuang-csm/hybrid-drt). The functions are also newer versions of the ones in the EIS package
- `xrd/`: functions for processing and plotting X-ray diffraction (XRD) data.
- `other_char/`: additional materials-characterization analysis and plotting routines.
- `global_params.py`: shared plotting parameters and styling defaults used across the package.
- `__init__.py`: package initialization.

- ## Purpose

The package reflects the analysis workflow behind my PhD research on proton-conducting ceramic fuel cells: taking raw instrument output, processing it consistently, and producing clean, readable, publication-grade figures. It is organized by characterization type so that functions can be reused across projects.

## Requirements

Python 3, with numpy, matplotlib, and the standard scientific-Python stack. [FILL IN: confirm any additional dependencies, for example pandas or an EIS-specific library.]

## Status

This is a work in progress and an evolving personal toolkit rather than a packaged release. It is research code, shared publicly for transparency.

## License

BSD-3-Clause.
