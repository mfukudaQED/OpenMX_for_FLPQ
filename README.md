# About
** UNDER CONSTRUCTION **

This is a modified OpenMX code for flpq calculations.  
This code can export density matrix including imagnary part  
and necessary parameters for post process calculations of local physical quantities. 
The original code of OpenMX is available at [https://www.openmx-square.org](https://www.openmx-square.org)

# Usage
The basic usage is the same as OpenMX.  
The following additional options are available.

## Options (Keywords)

- `DM.export`   off / on / window
  - on: Export density matrices DM, iDM and related parameters.
  - window: Export density matrices in a certain range of energy windows.
- `DM.specify.energy.range`  off (defalut) / on
  - `DM.energy.range  -2.0  0.0`  # default -10  10 in eV
  - `DM.energy.broadening.upper   0.001`  # default 300K in eV
  - `DM.energy.broadening.lower   0.001`  # default 300K in eV

# List of modified codes from original OpenMX

## Modified codes
- Search "FUKUDA" to identify the modified parts.

- DFT.c
  - The subroutines such as Band_DFT_Col_DMmu constructs density matrix DMmu in the range of certain energy window.
  - out_openmx_lpq exports the DM and parameters.

- Input_std.c
  - Added exporting density matrix parts.


- DFT.c, TRAN_DFT.c, tran_prototypes.c
  - iCDM and iDM are added even for collinear calculations to obtain kinetic momentum properly.

- makefile

## Additional codes
1. flpq_dm.h
    - Additional global parameters.

1. out_openmx_lpq.c
    - out_openmx_lpq exports the DM and parameters.

1. Allocate_DM_for_LPQ.c
1. Cluster_DFT_Col_DMmu.c
1. Cluster_DFT_NonCol_DMmu.c
1. Band_DFT_Col_DMmu.c
1. Band_DFT_NonCol_DMmu.c

# Contact
Masahiro FUKUDA (ISSP, Univ. of Tokyo)  
masahiro.fukuda__at__issp.u-tokyo.ac.jp  
Please replace __at__ by @.
