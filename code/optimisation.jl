# Termite mound radius and height optimisation.
#
# We ran six different optimisation algorithims to find the best radius and height with respect to the nest temperature.
# If the script is run normally, the optimisations starts only using the "generating_set_search" method, i.e. the "winning" algorithim. 
# The script can be executed from a termial in order to reproduce the results using the six different algorithims.
# Simply run "julia optimisation.jl 1", "julia optimisation.jl 2", "julia optimisation.jl 3", "julia optimisation.jl 4", "julia optimisation.jl 5", "julia optimisation.jl 6".
# Similar to the r-h-simulaitons this can be done in "parallel" on multiple instances.
# The methods are listed below (see line 12-17).

#  i  |  Optimisation Algorithim:
#-----------------------
#  1  |  separable_nes          |  28 steps | 0.869471955
#  2  |  xnes                   |  49 steps | 0.855152614
#  3  |  dxnes                  |  41 steps | 0.863387370
#  4  |  de_rand_1_bin          | 137 steps | 0.841589589
#  5  |  adaptive_de_rand_1_bin | 148 steps | 0.840296030
#  6  |  generating_set_search  |  77 steps | 0.838173109
#-----------------------

i = if abspath(PROGRAM_FILE) == @__FILE__
        if length(ARGS) < 1
            error("No argument provided.")
        end
        parse(Int, ARGS[1])
    else
        println("Optimisation stated.")
        print("Different solvers can used via: "); printstyled("run(", color = :yellow);printstyled("julia optimisation.jl i", color = :cyan);printstyled(")", color = :yellow);print(" for i ∈ {1, 2, 3, 4, 5, 6}\n\n")
        0
    end

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, OrdinaryDiffEqLowStorageRK, Interpolations, QuadGK, FastGaussQuadrature, Plots, BlackBoxOptim, Random
using Trixi: AbstractEquations, @muladd
import Interpolations: Line
import Trixi: flux_ranocha, ln_mean, inv_ln_mean, flux, varnames, cons2cons, cons2prim, prim2cons, cons2entropy, max_abs_speeds

include("src/equations/termite_mound_1d.jl")
include("src/callbacks/update_velocity_r-h.jl")

###############################################################################
# Output

out_dir = joinpath(@__DIR__, "out")
if isdir(out_dir) == false
    mkdir(out_dir)
end
output_dir = joinpath(out_dir, "optimisation")
if isdir(output_dir) == false
    mkdir(output_dir)
end

@inline function f_main(x0)
###############################################################################
# set parameters

r=x0[1]
h=x0[2]
g = 9.81
L = h + 0.3 + sqrt(0.09 + r*r) + sqrt(r*r + h*h)
p0 = 1e5
R = 8.314
M = 0.0289652
c_v = 20.85 
α_w = 1.1 
D0 = 0.05
Aᵣ = (D0 / 2)^2 * pi 
ρₕ₀ = 0.982 
η = 0.93 
uᵣ = 0.05 
ρᵣ = 1.17 
pᵣ = 1
xᵣ = L 
tᵣ = L/ uᵣ
xa = sqrt(0.09 + r*r) / L
xb = (sqrt(0.09 + r*r) + sqrt(r*r + h*h)) / L
xc = (L - 0.09) / L
xd = xc
Re = 9000 * 2 * sqrt(Aᵣ) / (xᵣ * sqrt(pi))
D = D0 / L
R = R / M
cᵥ = c_v / M
γ = (cᵥ + R) / cᵥ
Fr² = (uᵣ^2) / (g * L) 
𝑇ᵣ = pᵣ / (ρᵣ * R) 
λ_w = 64 / Re
β = (λ_w * xᵣ * sqrt(pi)) / (2 * 2 *sqrt(Aᵣ)) 
ε = (ρᵣ * (uᵣ^2)) / pᵣ  
k_w = (γ-1) * (xᵣ * α_w * 𝑇ᵣ * sqrt(pi)) / (sqrt(Aᵣ) * uᵣ * pᵣ) 
c_v = 790 
mass = (1742.448 * (1/3 * pi * r * r * h) ) ./ 8 
Li = (xᵣ*(xc-xa))
κ = 0.184 
k_i = tᵣ * κ / (c_v * mass/Li)
T_ref=297.76980029566033
t_ref=0.0033582989242263127
# set initial guess
T0 = 27.0
v0 = -2.219
Ti_LI = Get_initial_Ti(r,h)

###############################################################################
# Semidiscretization

equations = TermiteMoundEquations1D(;γ, k_i, k_w, tᵣ, uᵣ, xa, xb, xc, r, h, L, β, η, Fr², ρₕ₀, T_ref, t_ref, T0, v0, Ti_LI)

initial_condition = TermiteMoundInitialCondition
volume_flux = flux_ranocha
surface_flux = flux_ranocha

dg = DGSEM(polydeg = 4, surface_flux = flux_ranocha, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = TreeMesh((0.0,), (1.0,), initial_refinement_level = 5, n_cells_max = 1000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg, source_terms = source_terms, boundary_conditions = boundary_condition_periodic);

###############################################################################
# ODE solvers, callbacks etc.

tspan = (-0.1, 1.0).* 86400 ./ equations.tᵣ

ode = semidiscretize(semi, tspan)

stepsize_callback = StepsizeCallback(cfl = 0.25)
update_velocity_callback = UpdateVelocityCallback()

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = (u, equations) -> u[2]), base_level = 5, max_level = 6, max_threshold = -0.35)
amr_callback = AMRCallback(semi, amr_controller, interval = 1, adapt_initial_condition = true, adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(stepsize_callback, update_velocity_callback, amr_callback)

L2_error = try #polydeg=4, refinement=5, cfl=0.25, amr [≈400-600s] 
                sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
                            ode_default_options()..., callback = callbacks)
                if callbacks.discrete_callbacks[2].affect!.b() >2 
                    error("Increase resolution.")
                end
                callbacks.discrete_callbacks[2].affect!.e()
            catch
                Inf   
            end
    return L2_error
end

function f_test(x)
    x_sol = 0.88
    y_sol = 1.16
    return 20 + (x[1] - x_sol) * (x[1] - x_sol) + (x[2] - y_sol) * (x[2] - y_sol) - 10 * ( cos(2*pi*(x[1] - x_sol)) + cos(2*pi*(x[2] - y_sol)))
end

###############################################################################
### Optimisation ###

Random.seed!(10)
x0 = [0.74, 2.3666666666666667] # f_main(x0) = 0.8422245308178371
t_max = 2.0 * 24.0 * 3600.0
f_max = 100000
f_opt = f_main
#f_opt = f_test

opt =  if i == 1 
            print("Method:      ");printstyled("separable_nes\n", color = :cyan)
            bbsetup(f_opt; SearchRange = [(0.3, 1.5), (0.5, 2.5)], 
                    MaxTime = t_max, MaxFuncEvals = f_max,
                    TraceMode = :compact,
                    Method = :separable_nes)
        elseif i == 2
            print("Method:      ");printstyled("xnes\n", color = :cyan)
            bbsetup(f_opt; SearchRange = [(0.3, 1.5), (0.5, 2.5)], 
                    MaxTime = t_max, MaxFuncEvals = f_max,
                    TraceMode = :compact,
                    Method = :xnes)
        elseif i == 3
            print("Method:      ");printstyled("dxnes\n", color = :cyan)
            bbsetup(f_opt; SearchRange = [(0.3, 1.5), (0.5, 2.5)], 
                    MaxTime = t_max, MaxFuncEvals = f_max,
                    TraceMode = :compact,
                    Method = :dxnes)
        elseif i == 4
            print("Method:      ");printstyled("de_rand_1_bin\n", color = :cyan)
            bbsetup(f_opt; SearchRange = [(0.3, 1.5), (0.5, 2.5)], 
                    MaxTime = t_max, MaxFuncEvals = f_max,
                    TraceMode = :compact,
                    Method = :de_rand_1_bin)
        elseif i == 5
            print("Method:      ");printstyled("adaptive_de_rand_1_bin\n", color = :cyan)
            bbsetup(f_opt; SearchRange = [(0.3, 1.5), (0.5, 2.5)], 
                    MaxTime = t_max, MaxFuncEvals = f_max,
                    TraceMode = :compact,
                    Method = :adaptive_de_rand_1_bin)
        elseif i == 6 || i == 0
            print("Method:      ");printstyled("generating_set_search\n", color = :cyan)
            bbsetup(f_opt; SearchRange = [(0.3, 1.5), (0.5, 2.5)], 
                    MaxTime = t_max, MaxFuncEvals = f_max,
                    TraceMode = :compact,
                    Method = :generating_set_search)
        else
            error("Invalid input.")
        end

println("Max runtime: $(round(t_max/3600, digits=2)) h\n")
res = bboptimize(opt, x0) 
s = "#    Method:             $(res.method)
#    Termination reason: $(res.stop_reason)
#    x_opt: $(best_candidate(res))
#    f_opt: [$(best_fitness(res))]"

try 
    open(joinpath(output_dir, "output_$(i).jl"), "w") do file
    write(file, "## Optimisation $(i) ##\n#\n" * s * "\n#\n#------------------------------------------------------------------------------\n### read output ###\n@muladd @inline function read_output()\n    #\n    x_opt = $(best_candidate(res))\n    f_opt = $(best_fitness(res))\n    #\n    return x_opt, f_opt\nend\n#------------------------------------------------------------------------------\n")
    flush(file)
    end
catch
    open(joinpath(output_dir, "output_$(i).jl"), "w") do file
    write(file, "## output.jl ##\n#\n# optimisation\n#\n#------------------------------------------------------------------------------\n### read output ###\n@muladd @inline function read_output()\n    #\n    x_opt = NaN\n    f_opt = NaN\n    #\n    return x_opt, f_opt\nend\n#------------------------------------------------------------------------------\n")
    flush(file)
    end
end
