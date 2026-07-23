Number_of_Instances = 12

# The "Number_of_Instances" value can be changed depending on the availiable hardware.
# The script has to be executed from a termial opened in the same directory as this file.
# For the default case (Number_of_Instances = 12) one has to open 12 consoles.
# Enter one unique commands "julia r-h_simulations.jl 1", "julia r-h_simulations.jl 2", "julia r-h_simulations.jl 3", ... per instance.
# WAIT until all processes are done.
# Afterwards run the script "r-h_simulations_plots.jl" to create the png-files.

i = if abspath(PROGRAM_FILE) == @__FILE__
        if length(ARGS) < 1
            error("No argument provided.")
        end
        print("Simulations are running ...\n")
        println("Minimum runtime:  $(round(Int(ceil(961/Number_of_Instances)) * 400 /3600, digits=1)) hours")
        println("Maximum runtime: $(round(Int(ceil(961/Number_of_Instances)) * 1000 /3600, digits=1)) hours")
        parse(Int, ARGS[1])
    else
        printstyled("Warning: ", color = :yellow); print("This skript runs a total of 961 simulations.")
        print("\n         We recommend running multiple julia processes to distribute the workload.")
        print("\n         Currently the Number_of_Instances is set to $(Number_of_Instances).\n\n         Therefore: ")
        printstyled("run(", color = :yellow);printstyled("julia r-h_simulations.jl i", color = :cyan);printstyled(")", color = :yellow);print(" for i ∈ {1, 2, ..., $(Number_of_Instances)}\n\n")
        error("This error message (line 16) prevents the approximately $(round(961 * (0.5*(125+950)) /3600, digits=1)) hours computation.\n")
        0
    end

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, OrdinaryDiffEqLowStorageRK, Interpolations, QuadGK, FastGaussQuadrature, Plots
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
output_dir = joinpath(out_dir, "r-h_simulations")
if isdir(output_dir) == false
    mkdir(output_dir)
end

@inline function f_main(r,h)
###############################################################################
# set parameters

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
mesh2 = TreeMesh((0.0,), (1.0,), initial_refinement_level = 6, n_cells_max = 1000, periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg, source_terms = source_terms, boundary_conditions = boundary_condition_periodic);
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition, dg, source_terms = source_terms, boundary_conditions = boundary_condition_periodic);

###############################################################################
# ODE solvers, callbacks etc.

tspan = (-0.1, 1.0).* 86400 ./ equations.tᵣ

ode = semidiscretize(semi, tspan)
ode2 = semidiscretize(semi2, tspan)

stepsize_callback = StepsizeCallback(cfl = 0.25)
stepsize_callback2 = StepsizeCallback(cfl = 0.1)
update_velocity_callback = UpdateVelocityCallback()

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = (u, equations) -> u[2]), base_level = 5, max_level = 6, max_threshold = -0.35)
amr_callback = AMRCallback(semi, amr_controller, interval = 1, adapt_initial_condition = true, adapt_initial_condition_only_refine = true)

callbacks = CallbackSet(stepsize_callback, update_velocity_callback, amr_callback)
callbacks2 = CallbackSet(stepsize_callback, update_velocity_callback)
callbacks3 = CallbackSet(stepsize_callback2, update_velocity_callback)

L2_error, Vel_switches =try #polydeg=4, refinement=5, cfl=0.25, amr [≈400-600s] 
                            sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
                                        ode_default_options()..., callback = callbacks)
                            if callbacks.discrete_callbacks[2].affect!.e() > 2 || callbacks.discrete_callbacks[2].affect!.b() >2 
                                    error("Increase resolution.")
                            end
                            (callbacks.discrete_callbacks[2].affect!.e(), callbacks.discrete_callbacks[2].affect!.b())
                        catch
                            try #polydeg=4, refinement=6, cfl=0.25 [≈800-1000s] 
                                sol = solve(ode2, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
                                            ode_default_options()..., callback = callbacks2);
                                if callbacks.discrete_callbacks[2].affect!.e() > 2 || callbacks.discrete_callbacks[2].affect!.b() >2 
                                    error("Increase resolution.")
                                end
                                (callbacks.discrete_callbacks[2].affect!.e(), callbacks.discrete_callbacks[2].affect!.b())
                            catch
                                try #polydeg=4, refinement=6, cfl=0.1 [≈2000s] 
                                    sol = solve(ode2, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
                                                ode_default_options()..., callback = callbacks3);
                                    if callbacks.discrete_callbacks[2].affect!.e() > 2 || callbacks.discrete_callbacks[2].affect!.b() >2 
                                        error("Increase resolution.")
                                    end
                                    (callbacks.discrete_callbacks[2].affect!.e(), callbacks.discrete_callbacks[2].affect!.b())
                                catch
                                    (NaN, NaN)
                                end
                            end
                        end

string = "    r =  vcat(r, $(r) )\n    h = vcat(h, $(h) )\n    L2_error = vcat(L2_error, $(L2_error) )\n    Vel_switches = vcat(Vel_switches, $(Vel_switches) )\n    #\n"
open(joinpath(output_dir, "output_$(i).jl"), "a") do file
                write(file, string)
                flush(file)
end

    return nothing
end

function f_main_test(r,h)
    string = "r = vcat(r, $(r))\nh = vcat(h, $(h))\n\n"
        open(joinpath(output_dir, "output_$(i).jl"), "a") do file
            write(file, string)
            flush(file)
        end
    return nothing
end

###############################################################################
# Simulations

if i == 0
    procs = [run(`julia r-h_simulations.jl $i`) for i in 1:Number_of_Instances]
    include("r-h_simulations_plot.jl")
else
    string = "### output_$(i).jl ###\n#\n# r-h_simulations\n#\n#------------------------------------------------------------------------------\n### read output ###\n@muladd @inline function read_output_$(i)()\n    #\n    r = [0.0]\n    h = [0.0]\n    L2_error = [0.0]\n    Vel_switches = [0.0]\n    #\n"
        open(joinpath(output_dir, "output_$(i).jl"), "w") do file
            write(file, string)
            flush(file)
        end
    radius = range(0.3, 1.5, length=31)
    height = range(0.5, 2.5, length=31)
    N_total = 961
    N = Int(ceil(N_total/Number_of_Instances))
    for j = N*(i-1) + 1 : min(i*N, N_total)
        r = radius[mod1(j,31)]
        h = height[ceil(Int, j/31)]
        if j > 0
                open(joinpath(output_dir, "output_$(i).jl"), "a") do file
                write(file, "    # j = $j\n")
                flush(file)
            end
            f_main(r,h)
        end
    end  
end
