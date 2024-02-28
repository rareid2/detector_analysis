from simulation_engine import SimulationEngine
from hits import Hits
from scattering import Scattering
from deconvolution import Deconvolution
from scipy import interpolate
from macros import find_disp_pos
import numpy as np
import os
from scipy.io import savemat
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotting.plot_settings import *
import copy

# construct = CA and TD
# source = DS and PS

import subprocess

def read_csv_in_parts(base_fname, fname_tag, simulation_engine, n_elements_original, energy_level, multiplier: int = 1):
    # Replace 'your_file.csv' with the actual path to your CSV file
    file_path = base_fname

    # Construct the terminal command using the wc command
    command = ['wc', '-l', file_path]

    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        # Extract the number of lines from the command output
        num_lines = int(result.stdout.split()[0])
        print(f'The number of lines in {file_path} is: {num_lines}')
    else:
        print(f'Error running command: {result.stderr}')

    # now we run through the lines
    sections = num_lines // 20
    raw_hits = np.zeros((n_elements_original*multiplier, n_elements_original*multiplier))
    for ti, ii in enumerate(range(sections, num_lines, sections)):

        myhits = Hits(fname=base_fname, experiment=False, txt_file=False, nlines=sections, nstart=ii)
        myhits.get_det_hits(
            remove_secondaries=False, second_axis="y", energy_level=energy_level
        )

        # directory to save results in
        results_dir = "../simulation-results/rings/"
        results_tag = f"{fname_tag}_{ii}"
        results_save = results_dir + results_tag

        # deconvolution steps
        deconvolver = Deconvolution(myhits, simulation_engine)

        deconvolver.deconvolve(
            downsample=int(n_elements_original*multiplier),
            trim=None,
            vmax=None,
            plot_deconvolved_heatmap=False,
            plot_raw_heatmap=False,
            save_raw_heatmap = results_save + "_raw.png",
            plot_signal_peak=False,
            plot_conditions=False,
            flat_field_array=None,
            hits_txt=False,
            rotate=True,
            delta_decoding=True,
            apply_noise=False,
        )

        raw_hits += np.loadtxt(results_save+"_raw.txt")

    return raw_hits
