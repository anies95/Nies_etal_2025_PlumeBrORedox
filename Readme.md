# Combustion-Atmospheric Volcanic Plume Chemistry and Transport Model
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15068003.svg)](https://doi.org/10.5281/zenodo.15068003)
developed and applied by Nies et al., 2025, "Reactive bromine in volcanic plumes confines the emission temperature and oxidation of magmatic gases at the atmospheric interface" 

high temperature (HT) stage based on Kuhn et al., 2022, https://doi.org/10.1029/2022GC010671  
based on the cantera toolkit (cantera.org): Goodwin et al., 2024, https://doi.org/10.5281/zenodo.14455267  
chemical mechanisms (HT & ATM stage) are described in Nies, 2024, https://theses.hal.science/tel-04973866  

author contacts:  
Alexander Nies, email: alexander.nies0795@gmail.com   
Jonas Kuhn, email: jonaskuhn-science@posteo.de  
Tjarda J. Roberts, email: tjarda.roberts@lmd.ipsl.fr

# Overview
This model code allows to simulate the chemical evolution of a volcanic plume from the high-temperature emission to the atmosphere (HT stage)
until the first hours of downwind evolution (ATM stage) alongside a physical mixing and dilution trajectory. All relevant files to run the model
are included in this repository, including the chemical mechanisms for the two stages.

# Prerequisites
Python
required packages: NumPy, SciPy, Os, Sys, Pandas, Cantera

# Example
Nies_etal_2025_example_model_run.py includes an exemplary script running both model stages and saving the output in Python dictionaries for further data processing.

# Scientific Use and Collaboration
We encourage the scientific community to use and build upon this work. If you use this code in your research, please cite our work accordingly.
Citation guidelines and licensing terms can be found in the LICENSE file included in this repository.
We welcome collaboration and feedback to improve the project further. Feel free to reach out with suggestions, contributions, or discussion opportunities.
