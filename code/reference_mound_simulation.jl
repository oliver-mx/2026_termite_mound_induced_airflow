# 24h air flow simulation of the reference mound (King et al., 2015)
# 
# Runtime: ≈ 650s
# 

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, OrdinaryDiffEqLowStorageRK, Interpolations, QuadGK, FastGaussQuadrature, Plots
using Trixi: AbstractEquations, @muladd
import Interpolations: Line
import Trixi: flux_ranocha, ln_mean, inv_ln_mean, flux, varnames, cons2cons, cons2prim, prim2cons, cons2entropy, max_abs_speeds

include("src/equations/termite_mound_1d.jl")
include("src/callbacks/update_velocity_reference.jl")

###############################################################################
# Output

out_dir = joinpath(@__DIR__, "out")
if isdir(out_dir) == false
    mkdir(out_dir)
end
output_dir = joinpath(out_dir, "reference_mound")
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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
            ode_default_options()..., callback = callbacks);

    return (callbacks.discrete_callbacks[2].affect!.e(), callbacks.discrete_callbacks[2].affect!.b())
end

###############################################################################
# Simulation

t_start = time()
(ϵ, ζ) = f_main(0.6,2.0)
t_end = time() - t_start

println("(ϵ, ζ) = ($ϵ, $ζ)")
println("Simulation time: $(round(t_end,digits=1))s")

include("src/plots/reference_mound_plot.jl")
