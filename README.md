# Learning from modelling and simulation of air flow dynamics inside termite mounds in view of low-energy houses

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://...)](https://doi.org/...)


This repository contains information and code to reproduce the results presented in the article
```bibtex
@online{marx26,
  title={Learning from modelling and simulation of air flow dynamics inside termite mounds in view of low-energy houses},
  author={Marx, Oliver P and Gasser, Ingenuin and Schmidgall, Annika},
  year={2026},
  month={},
  journal={...},
  volume={...},
  number={...},
  pages={...--...},
  publisher={...}, 
  doi = {https://doi.org/... }
}
```


## Abstract

Termite mounds are complex animal-built structures with varied architectural designs depending on the species and respective habitats.
One key function is to provide ventilation for the underground nest.
This regulates the temperature of the nest and diffuses the metabolic gases produced inside.
A one-dimensional mathematical model is presented that describes the airflow inside of a termite mound.
This thermo fluid dynamic model originates from the Euler equations of gas dynamics in a low Mach number regime.
Numerical simulations are performed to study full 24 hour day-night air flow cycle and the occurrence of day-night air flow cycles for differently sized termite mounds.
An optimal mound geometry is determined, which achieves stable nest temperature control.
This model acts as a blueprint for the modeling, simulation and optimisation of modern low-energy or passive buildings where the supporting air ventilation system is crucial.
As an example we adapt the model for a certain type of passive house.
A 150 day winter scenario is simulated for a low tech passive house design.


## Numerical experiments

In order to reproduce the numerical experiments presented in this article, you need to install [Julia](https://julialang.org/). 
The numerical experiments presented in this article were performed using Julia v1.12.6.
Download this repository, e.g., by cloning it with `git` or by downloading an archive via the GitHub interface.
Then, start Julia in the `code` directory of this repository and follow the instructions described in the `README.md` file therein.


## Authors

- Oliver P Marx (University of Hamburg, Germany)
- Ingenuin Gasser (University of Hamburg, Germany)
- Annika Schmidgall (University of Hamburg, Germany)


## License

The code in this repository is published under the MIT license, see the `LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!