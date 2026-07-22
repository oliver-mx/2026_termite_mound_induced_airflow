# Simulates 150 days of the passive house design from Pak et al. (2009).
# Air and soil temperature are similar to: Oktober 2024 - March 2025 weather data from Hamburg, Germany.
# 
# Runtime: ≈ 20000s
# 

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, OrdinaryDiffEqLowStorageRK, Interpolations, QuadGK, FastGaussQuadrature, Plots
using Trixi: AbstractEquations, @muladd
import Interpolations: Line
import Trixi: flux_ranocha, ln_mean, inv_ln_mean, flux, varnames, cons2cons, cons2prim, prim2cons, cons2entropy, max_abs_speeds

include("src/equations/winter_1d.jl")
include("src/callbacks/update_velocity_winter.jl")

###############################################################################
# Output

out_dir = joinpath(@__DIR__, "out")
if isdir(out_dir) == false
    mkdir(out_dir)
end
output_dir = joinpath(out_dir, "winter_simulation")
if isdir(output_dir) == false
    mkdir(output_dir)
end

@inline function f_main(x0, x1, x2, x3, x4, x5)
###############################################################################
# set parameters

g = 9.81
L = 21.5
h = 5.0
p0 = 1e5
R = 8.314
M = 0.0289652
c_v = 20.85 
α_w = 1.1 
D0 = 0.6207042780583918
Aᵣ = 3.9
ρₕ₀ = 0.982 
η = 0.93 
uᵣ = 0.2 
ρᵣ = 1.17 
pᵣ = 1
xᵣ = L 
tᵣ = L/ uᵣ
xa = 3/L
xb = 8/L
xc = 1.0 - xa 
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
c_v = 750
mass = 10000.0
Li = xᵣ*(xc-xa)
κ = 0.1
k_i = tᵣ * κ / (c_v * mass/Li)
α_s = 0.3
Ls = xᵣ*(xb-xa)
W_s = 1.0
q_r = 200
q₀ = q_r
k_s = (tᵣ * α_s * W_s/Ls * q_r) / (c_v * mass/Li * (𝑇ᵣ*p0))
T_ref = 297.76980029566033
t_ref = 0.0033582989242263127
# set initial guess
T0 = 27.0
v0 = -2.219
Ti_LI = LinearInterpolation([0.0, 0.372, 0.373, 1.0],[23.8, 24.0, 18.0, 17.5])
nodes = [0.0, 0.37068966, 0.74137931, 1.11206897, 1.48275862, 1.85344828, 2.22413793, 2.59482759, 2.96551724, 3.3362069, 3.70689655, 4.07758621, 4.44827586, 4.81896552, 5.18965517, 5.56034483, 5.93103448, 6.30172414, 6.67241379, 7.04310345, 7.4137931, 7.78448276, 8.15517241, 8.52586207, 8.89655172, 9.26724138, 9.63793103, 10.00862069, 10.37931034, 10.75, 11.12068966, 11.49137931, 11.86206897, 12.23275862, 12.60344828, 12.97413793, 13.34482759, 13.71551724, 14.0862069, 14.45689655, 14.82758621, 15.19827586, 15.56896552, 15.93965517, 16.31034483, 16.68103448, 17.05172414, 17.42241379, 17.79310345, 18.1637931, 18.53448276, 18.90517241, 19.27586207, 19.64655172, 20.01724138, 20.38793103, 20.75862069, 21.12931034, 21.5] ./ 21.5
ρ_LI = LinearInterpolation(nodes, x1)
v_LI = LinearInterpolation(nodes, x2)
T_LI = LinearInterpolation(nodes, x3)
Ti_LI = LinearInterpolation(nodes, x4)
p0 = x5[]

###############################################################################
# Semidiscretization

equations = PassiveHouseEquations1D(;γ, k_i, k_w, k_s, q₀, tᵣ, uᵣ, xa, xb, xc, h, L, β, η, Fr², ρₕ₀, T_ref, t_ref, ρ_LI, v_LI, T_LI, Ti_LI, p0)

initial_condition = PassiveHouseInitialCondition
volume_flux = flux_ranocha
surface_flux = flux_ranocha

dg = DGSEM(polydeg = 4, surface_flux = flux_ranocha, volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = TreeMesh((0.0,), (1.0,), initial_refinement_level = 6, n_cells_max = 1000, periodicity = true) 

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg, source_terms = source_terms);

###############################################################################
# ODE solvers, callbacks etc.

tspan = x0 .* 86400 ./ equations.tᵣ
ode = semidiscretize(semi, tspan)

stepsize_callback = StepsizeCallback(cfl = 1.2)
update_velocity_callback = UpdateVelocityCallback()

callbacks = CallbackSet(stepsize_callback, update_velocity_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false); dt = 1.0,
        ode_default_options()..., callback = callbacks)

    y1 = callbacks.discrete_callbacks[2].affect!.rho_s
    y2 = callbacks.discrete_callbacks[2].affect!.v1_s
    y3 = callbacks.discrete_callbacks[2].affect!.T_s
    y4 = callbacks.discrete_callbacks[2].affect!.Ti_s
    y5 = callbacks.discrete_callbacks[2].affect!.p0_s
    y6 = callbacks.discrete_callbacks[2].affect!.Tu_s
    y7 = callbacks.discrete_callbacks[2].affect!.T_inside_s

    return y1, y2, y3, y4, y5, y6, y7
end

function f_main_test(x0, x1, x2, x3, x4, x5) # only for testing
    return x1 .+ .1, x2, 1.1 .* x3, 1.01 .* x4, [1.0], x1, [sum([x for x in x3])./length(x3)]
end

###############################################################################
# Simulation

t_start = time()
t = vcat([-0.03], 0:1:150)
f = f_main
#f = f_main_test
#-----------------------
rho_prev = Ref(zeros(59,))
u1_prev = Ref(zeros(59,))
T_prev = Ref(zeros(59,))
Ti_prev = Ref(zeros(59,))
p0_prev = Ref([0.0])
Tu_prev = Ref(zeros(59,))
T_inside_prev = Ref([0.0])
#-----------------------
### Loop ###

for i = 3:2:152
    y1, y2, y3, y4, y5, y6, y7 = if i == 3
                                y1, y2, y3, y4, y5, y6, y7 = f((t[i-2], t[i-1]), [1.16477, 1.16462, 1.16445, 1.1642, 1.16386, 1.16335, 1.16289, 1.1622, 1.16157, 1.16136, 1.16121, 1.16087, 1.16051, 1.16021, 1.15957, 1.15906, 1.15833, 1.15745, 1.15638, 1.15509, 1.15358, 1.15177, 1.15163, 1.15176, 1.1519, 1.15195, 1.15212, 1.15223, 1.15233, 1.1524, 1.15261, 1.15272, 1.15288, 1.15307, 1.1533, 1.15355, 1.154, 1.15423, 1.15467, 1.15528, 1.15602, 1.15671, 1.15736, 1.1585, 1.15948, 1.16052, 1.16179, 1.16314, 1.16459, 1.16601, 1.16658, 1.16701, 1.16578, 1.16569, 1.16545, 1.16508, 1.1649, 1.16475, 1.16464],
                                [0.04011, 0.04012, 0.04016, 0.04025, 0.04051, 0.04118, 0.04258, 0.04557, 0.05063, 0.05747, 0.06386, 0.06817, 0.07052, 0.07167, 0.07227, 0.07259, 0.07289, 0.07335, 0.07435, 0.07647, 0.08111, 0.09011, 0.10366, 0.11915, 0.13166, 0.13981, 0.14359, 0.1455, 0.1463, 0.1466, 0.14664, 0.14644, 0.14584, 0.14454, 0.1413, 0.13505, 0.12419, 0.10902, 0.09419, 0.0836, 0.07782, 0.07492, 0.07366, 0.07301, 0.07257, 0.07198, 0.0709, 0.06866, 0.06452, 0.05832, 0.05164, 0.04631, 0.04309, 0.04153, 0.04073, 0.04039, 0.04022, 0.04014, 0.04011],
                                [13.51432, 13.55001, 13.59183, 13.65334, 13.73902, 13.86413, 13.97778, 14.14859, 14.30436, 14.35578, 14.39243, 14.47745, 14.56509, 14.63957, 14.79837, 14.92579, 15.10815, 15.32598, 15.59283, 15.91619, 16.29503, 16.74828, 16.78453, 16.75163, 16.71745, 16.70391, 16.66135, 16.63247, 16.60915, 16.59195, 16.53819, 16.50953, 16.47032, 16.42248, 16.36378, 16.3015, 16.18984, 16.13159, 16.01995, 15.86797, 15.68308, 15.51107, 15.34827, 15.0653, 14.82159, 14.56308, 14.24885, 13.91659, 13.5581, 13.20901, 13.06951, 12.96359, 13.26582, 13.28704, 13.34605, 13.43672, 13.48134, 13.5179, 13.54678],
                                vcat(27.0 .* ones(22,), 18.0 .* ones(37,)), [1.0])
                                else
                                rho_prev[], u1_prev[], T_prev[], Ti_prev[], p0_prev[], Tu_prev[], T_inside_prev[]
                            end
    if i == 3
        println("Day 0")
        string1 = createString1([0.0, 0.37068966, 0.74137931, 1.11206897, 1.48275862, 1.85344828, 2.22413793, 2.59482759, 2.96551724, 3.3362069, 3.70689655, 4.07758621, 4.44827586, 4.81896552, 5.18965517, 5.56034483, 5.93103448, 6.30172414, 6.67241379, 7.04310345, 7.4137931, 7.78448276, 8.15517241, 8.52586207, 8.89655172, 9.26724138, 9.63793103, 10.00862069, 10.37931034, 10.75, 11.12068966, 11.49137931, 11.86206897, 12.23275862, 12.60344828, 12.97413793, 13.34482759, 13.71551724, 14.0862069, 14.45689655, 14.82758621, 15.19827586, 15.56896552, 15.93965517, 16.31034483, 16.68103448, 17.05172414, 17.42241379, 17.79310345, 18.1637931, 18.53448276, 18.90517241, 19.27586207, 19.64655172, 20.01724138, 20.38793103, 20.75862069, 21.12931034, 21.5],
                      t[i-1], y1, y2, y3, y4, y5[], y6, y7[])
        open(joinpath(output_dir, "output.jl"), "w") do file
            write(file, string1)
            flush(file)
        end
    end                   
    # z-vals at t[i] (simulated)
    z1, z2, z3, z4, z5, z6, z7 = f((t[i-1], t[i]), y1, y2, y3, y4, y5[])
    println("Day $(i-2)")
    string2 = createString2([t[i]], z1, z2, z3, z4, z5[], z6, z7[])
    open(joinpath(output_dir, "output.jl"), "a") do file
        write(file, "    # simulated \n" * string2)
        flush(file)
    end
    # q-vals at t[i+1] (interpolated)
    q1, q2, q3, q4, q5, q6, q7 = NextVals(t[i-1:i+1], z1, z2, z3, z4, z5, z6, y7, z7)
    rho_prev[] = q1; u1_prev[] = q2; T_prev[] = q3; Ti_prev[] = q4; p0_prev[] = q5; Tu_prev[] = q6; T_inside_prev[] = q7
    println("Day $(i-1)")
    string3 = createString2([t[i+1]], q1, q2, q3, q4, q5[], q6, q7[])
    open(joinpath(output_dir, "output.jl"), "a") do file
        write(file, "    # interpolated \n" * string3)
        flush(file)
    end
end

open(joinpath(output_dir, "output.jl"), "a") do file
    write(file, "    return nodes, t, rho, v, T, Ti, Tu, p0, Tinside\nend\n#------------------------------------------------------------------------------\n")
    flush(file)
end

t_end = time() - t_start
println("Simulation time: $(round(t_end,digits=1))s")
include("src/plots/winter_plot.jl")
