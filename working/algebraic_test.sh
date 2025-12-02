#!/bin/bash

lut_path='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/lut_full/lut.nc'
rdn_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_rdn_b0106_v01.img'
obs_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_obs_b0106_v01.img'
loc_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_loc_b0106_v01.img'
atm_file='/Users/bgreenbe/Github/isofit-jax/working/heuristic_test'
out_file='/Users/bgreenbe/Github/isofit-jax/working/algebraic_test'

python cli_inversion.py algebraic $lut_path $rdn_file $obs_file $loc_file $atm_file $out_file
