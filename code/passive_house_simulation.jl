# 24h air flow simulation of the passive house design from Pak et al. (2009)
# 
# Runtime: ≈ 280s
# 

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, OrdinaryDiffEqLowStorageRK, Interpolations, QuadGK, FastGaussQuadrature, Plots
using Trixi: AbstractEquations, @muladd
import Interpolations: Line
import Trixi: flux_ranocha, ln_mean, inv_ln_mean, flux, varnames, cons2cons, cons2prim, prim2cons, cons2entropy, max_abs_speeds

include("src/equations/passive_house_1d.jl")
include("src/callbacks/update_velocity_passive_house.jl")

###############################################################################
# Output

out_dir = joinpath(@__DIR__, "out")
if isdir(out_dir) == false
    mkdir(out_dir)
end
output_dir = joinpath(out_dir, "passive_house")
if isdir(output_dir) == false
    mkdir(output_dir)
end

@inline function f_main(x_input)
###############################################################################
# Semidiscretization

equations = PassiveHouseEquations1D()

initial_condition = PassiveHouseInitialCondition
volume_flux = flux_ranocha
surface_flux = flux_ranocha

dg = DGSEM(polydeg = 4, surface_flux = flux_ranocha, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = TreeMesh((0.0,), (1.0,), initial_refinement_level = 6, n_cells_max = 1000, periodicity = true) 

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg, source_terms = source_terms);

###############################################################################
# ODE solvers, callbacks etc.

tspan = (-0.1, 1.0).* 86400 ./ equations.tᵣ

ode = semidiscretize(semi, tspan)

stepsize_callback = StepsizeCallback(cfl = 2.0)
update_velocity_callback = UpdateVelocityCallback()

callbacks = CallbackSet(stepsize_callback, update_velocity_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
            ode_default_options()..., callback = callbacks);

    return callbacks.discrete_callbacks[2].affect!.a()
end

###############################################################################
# Simulation

t_start = time()
f_main(0.0)
t_end = time() - t_start
println("Simulation time: $(round(t_end,digits=1))s")

include("src/plots/passive_house_plot.jl")
