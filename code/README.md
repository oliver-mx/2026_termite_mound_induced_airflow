# Numerical Experiments

In order to reproduce the numerical experiments presented in this article, you need to install [Julia](https://julialang.org/).
Start Julia (v1.12.6) in the `code` directory of this repository.

The `code` directory has the subdirectories `out` and `src`.
All output files of executed scripts are saved inside the `out` directory.


The following List describes which script creates which figure(s).
* Figures 2(a), 2(b): `r-h_simulations_plot.jl`
* Figures 3(a), 3(b), 3(c), 3(d), 4(a), 4(b): `optimal_mound_simulation.jl`
* Figures 6(a), 6(b), 7(a), 7(b): `winter_simulation.jl`

The resulting figures are then saved as .png files in the respective folder within the `out` directory.

Now follows a basic explanation of all scripts inside the `code` directory in alphabetical order.

The file `optimal_mound_simulation.jl` simulates a full day of a termite mound with optimal radius and height.
The optimal values are already inserted but can be obtained from `optimisation.jl`. 
Run

```julia
julia> include("optimal_mound_simulation.jl")
```

The file `optimisation.jl` searches for an optimal termite mound geometry using optimisation algorithims implemented in [BlackBoxOptim.jl.jl](https://github.com/SciML/BlackBoxOptim.jl).
Different optimisation Methods have been applied to our minimisation problem.
The best performing method is `separable_nes`.
It is set to the default, when executing

```julia
julia> include("optimisation.jl")
```

The other methods can be called when running the file with an input argument `i ∈ {1,2,3,4,5,6}`.
For example let us consider `i = 4`.
Open a terminal in the `code` directory and run

```julia
bash> julia optimisation.jl 4
```

Similar to `optimal_mound_simulation.jl` the file `passive_house_simulation.jl` simulates a full day of the passive house presented in Pak et al. (2009).

```julia
julia> include("passive_house_simulation.jl")
```

Likewise, in `reference_mound_simulation.jl` a 24h simulation of the reference termite mound (King et al., 2015) is implemented.

```julia
julia> include("reference_mound_simulation.jl")
```

Next, `r-h_simulations.jl` simulates full day-night cycles of 961 differently sized termite mounds.  
Due to the large number of simulations, it is intended to split the workload across multiple julia instances.
By default `Number_of_Instances = 12` (see line 1 of `r-h_simulations.jl`).

In order to reproduce all results, one has to open `12` consoles in the `code` directory.
Each console opens the script `r-h_simulations.jl` in julia while providing a unique input argument of the set `{1,2,...,Number_of_Instances}`.
For example, worker 1 executes

```julia
bash> julia r-h_simulations.jl 1
```

Next, instance 2 performs

```julia
bash> julia r-h_simulations.jl 2
```

Finally, the last worker runs

```julia
bash> julia r-h_simulations.jl 12
```

Once all 961 simulations are complete one can plot the results by executing

```julia
julia> include("r-h_simulations_plot.jl")
```

The last file `winter_simulation.jl` simulates the 150 day winter scenario.

```julia
julia> include("winter_simulation.jl")
```