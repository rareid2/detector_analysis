# script to estiamte activity of Cd 109
import numpy as np 
from scipy import interpolate
from datetime import date, datetime

from experiment_constants import *

# function to read NIST table of data
def read_NIST_tabe(fname, rho):
    file1 = open(fname, 'r')
    Lines = file1.readlines()

    count = 0
    energies = []
    mu = []
    mu_en = []

    # extract data from file
    for line in Lines:
        count += 1
        # ignore header
        if count > 4:
            line_split = line.split(' ')

            # save energy and range
            energies.append(float(line_split[0]))
            mu.append(float(line_split[2])*rho)
            mu_en.append(float(line_split[4])*rho)
    
    # convert energy
    energies = np.array(energies)*1000 # convert to keV

    # mu_en will be in inverse cm

    return energies, mu, mu_en

# function to estimate the QDE given a known source activity
def estimate_qde(counts,time,emission_peaks,branching_ratios,distance,activity_bq,activity_today_unc):

    energies, mu, mu_en = read_NIST_tabe("../experiment_results/xray_mass_att_air.txt", rho_air)
    energies_plastic, mu_plastic, mu_en_plastic = read_NIST_tabe("../experiment_results/xray_mass_att_polymethyl_methacrylate.txt" , rho_pla)

    air_att_weighted_avg = 0
    plastic_att_weighted_avg = 0

    # interpolate to get mu en
    for emission_peak, branching_ratio in zip(emission_peaks,branching_ratios):

        interp_f_air = interpolate.interp1d(energies, mu_en)
        interp_f_plastic = interpolate.interp1d(energies_plastic, mu_en_plastic)

        mu_ene = interp_f_air(emission_peak)
        mu_ene_plastic = interp_f_plastic(emission_peak)

        # find the attenuation at the set distance
        att_coef = np.exp(-mu_ene * distance)
        att_coef_plastic = np.exp(-mu_ene_plastic * plastic_distance)

        # weighted average if more than one
        if len(emission_peaks) > 1:
            air_att_weighted_avg += att_coef*(branching_ratio / (sum(branching_ratios)))
            plastic_att_weighted_avg += att_coef_plastic*(branching_ratio / (sum(branching_ratios)))
        else:
            air_att_weighted_avg += att_coef
            plastic_att_weighted_avg += att_coef_plastic

    # attenuation due to spreading (solid angle coverage)
    dist_att = (1/2) - (2/np.pi) * np.arctan(1 / np.sqrt( 1 + (0.5 * (detector_side_length_cm/distance)**2) ))
    dist_att_unc = distance_uncertainty * (3.97887 / ( np.sqrt( (12.5 / distance**2) + 1) * ( distance**3 +(6.25*distance)) ))

    counts_unc = np.sqrt(counts)
    # total branching
    branching = sum(branching_ratios)

    denom = time * activity_bq * air_att_weighted_avg * plastic_att_weighted_avg * branching * dist_att
    qde = 100* counts / denom
    uncertainty = 100*np.sqrt( ((1/denom)**2 * counts_unc**2)  + ((-1* counts / denom / dist_att)**2 * dist_att_unc**2) + ((-1*counts / (denom * activity_bq))**2 * activity_today_unc**2) )

    return qde, uncertainty

# function to estimate the QDE given a known source activity
def estimate_activity(counts,time,emission_peaks,branching_ratios,distance,qde):

    energies, mu, mu_en = read_NIST_tabe("../experiment_results/xray_mass_att_air.txt", rho_air)
    energies_plastic, mu_plastic, mu_en_plastic = read_NIST_tabe("../experiment_results/xray_mass_att_polymethyl_methacrylate.txt" , rho_pla)

    air_att_weighted_avg = 0
    plastic_att_weighted_avg = 0

    # interpolate to get mu en
    for emission_peak, branching_ratio in zip(emission_peaks,branching_ratios):

        interp_f_air = interpolate.interp1d(energies, mu_en)
        interp_f_plastic = interpolate.interp1d(energies_plastic, mu_en_plastic)

        mu_ene = interp_f_air(emission_peak)
        mu_ene_plastic = interp_f_plastic(emission_peak)

        # find the attenuation at the set distance
        att_coef = np.exp(-mu_ene * distance)
        att_coef_plastic = np.exp(-mu_ene_plastic * plastic_distance)

        # weighted average if more than one
        if len(emission_peaks) > 1:
            air_att_weighted_avg += att_coef*(branching_ratio / (sum(branching_ratios)))
            plastic_att_weighted_avg += att_coef_plastic*(branching_ratio / (sum(branching_ratios)))
        else:
            air_att_weighted_avg += att_coef
            plastic_att_weighted_avg += att_coef_plastic

    # attenuation due to spreading (solid angle coverage)
    dist_att = (1/2) - (2/np.pi) * np.arctan(1 / np.sqrt( 1 + (0.5 * (detector_side_length_cm/distance)**2) ))
    dist_att_unc = distance_uncertainty * (3.97887 / ( np.sqrt( (12.5 / distance**2) + 1) * ( distance**3 +(6.25*distance)) ))

    counts_unc = np.sqrt(counts)
    # total branching
    branching = sum(branching_ratios)

    denom = time * qde * air_att_weighted_avg * plastic_att_weighted_avg * branching * dist_att
    activity_bq = counts / denom
    uncertainty = np.sqrt( ((1/denom)**2 * counts_unc**2)  + ((-1* counts / denom / dist_att)**2 * dist_att_unc**2) )

    return activity_bq, uncertainty


# function to estimate the QDE given a known source activity
def estimate_counts(emission_peaks,branching_ratios,distance,qde, activity_bq):

    energies, mu, mu_en = read_NIST_tabe("../experiment_results/xray_mass_att_air.txt", rho_air)
    energies_plastic, mu_plastic, mu_en_plastic = read_NIST_tabe("../experiment_results/xray_mass_att_polymethyl_methacrylate.txt" , rho_pla)

    air_att_weighted_avg = 0
    plastic_att_weighted_avg = 0

    # interpolate to get mu en
    for emission_peak, branching_ratio in zip(emission_peaks,branching_ratios):

        interp_f_air = interpolate.interp1d(energies, mu_en)
        interp_f_plastic = interpolate.interp1d(energies_plastic, mu_en_plastic)

        mu_ene = interp_f_air(emission_peak)
        mu_ene_plastic = interp_f_plastic(emission_peak)

        # find the attenuation at the set distance
        att_coef = np.exp(-mu_ene * distance)
        att_coef_plastic = np.exp(-mu_ene_plastic * plastic_distance)

        # weighted average if more than one
        if len(emission_peaks) > 1:
            air_att_weighted_avg += att_coef*(branching_ratio / (sum(branching_ratios)))
            plastic_att_weighted_avg += att_coef_plastic*(branching_ratio / (sum(branching_ratios)))
        else:
            air_att_weighted_avg += att_coef
            plastic_att_weighted_avg += att_coef_plastic

    # attenuation due to spreading (solid angle coverage)
    dist_att = (1/2) - (2/np.pi) * np.arctan(1 / np.sqrt( 1 + (0.5 * (detector_side_length_cm/distance)**2) ))
    dist_att_unc = distance_uncertainty * (3.97887 / ( np.sqrt( (12.5 / distance**2) + 1) * ( distance**3 +(6.25*distance)) ))

    # total branching
    branching = sum(branching_ratios)

    denom = qde * air_att_weighted_avg * plastic_att_weighted_avg * branching * dist_att
    counts  = denom * activity_bq
    counts_unc = np.sqrt(counts)

    uncertainty = np.sqrt( ((1/denom)**2 * counts_unc**2)  + ((-1* counts / denom / dist_att)**2 * dist_att_unc**2) )

    return counts, uncertainty

def get_activity(activity_bq, activity_date, half_life, half_life_unc):
    # get activity on the day teeest was taken
    test_date = datetime(2023,1,25)
    d1 = date(test_date.year,test_date.month,test_date.day)

    decay_constant = np.log(2)/half_life
    decay_constant_unc = -1*(half_life_unc) * np.log(2)/half_life**2
    delta_days = d1 - activity_date
    delta_days = delta_days.days

    activity_today_bq = activity_bq * np.exp(-1*decay_constant * delta_days)
    activity_today_unc = -1* delta_days * activity_bq * np.exp(-1*decay_constant * delta_days) * decay_constant_unc

    return activity_today_bq, activity_today_unc

def read_qde_curve(fname):
    file1 = open(fname, 'r')
    Lines = file1.readlines()

    count = 0
    energies = []
    qde = []

    # extract data from file
    for line in Lines:
        count += 1
        # ignore header
        if count > 1:
            line_split = line.split(' ')
    
            # save energy and range
            energies.append(float(line_split[0].strip(',')))
            qde.append(float(line_split[1]))
    
    file1.close()
    return energies,qde