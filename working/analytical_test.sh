#!/bin/bash

config_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/config/emit20220818t205752_isofit.json'
lut_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock_6c/lut_full/lut.nc'
rdn_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_rdn_b0106_v01.img'
obs_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_obs_b0106_v01.img'
loc_file='/Users/bgreenbe/Projects/AOD/emit20220818t205752_blackrock/input/emit20220818t205752_o23014_s000_l1b_loc_b0106_v01.img'
x0_file='/Users/bgreenbe/Github/isofit-jax/working/algebraic_rfl'
atm_file='/Users/bgreenbe/Github/isofit-jax/working/heuristic_atm'
out_file='/Users/bgreenbe/Github/isofit-jax/working/analytical_rfl'
sub_file='/Users/bgreenbe/Github/isofit-jax/working/algebraic_rfl'
batchsize=200

python cli_inversion.py analytical $config_file $lut_file $rdn_file $obs_file \
        $loc_file $x0_file $atm_file $out_file $sub_file --batchsize $batchsize
