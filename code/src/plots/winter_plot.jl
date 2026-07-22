# Creates plots of the `winter_simulation.jl` script
# 

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, Plots, Trixi2Vtk, Interpolations, OrdinaryDiffEq, QuadGK, FastGaussQuadrature
using Trixi: AbstractEquations, @muladd
import Interpolations: Line
import Trixi: flux_ranocha, ln_mean, inv_ln_mean, flux, varnames, cons2cons, cons2prim, prim2cons, cons2entropy, max_abs_speeds

###############################################################################
# Output
out_dir = joinpath(@__DIR__, "../../out")
output_dir = joinpath(out_dir, "winter_simulation")
if isdir(output_dir) == false
    error("Run winter_simulation.jl first!")
end
plot_dir = joinpath(output_dir, "plots")
if isdir(plot_dir) == false
    mkdir(plot_dir)
end

#------------------------------------------------------------------------------
include("../equations/winter_1d.jl")
include("../callbacks/update_velocity_winter.jl")
include("../../out/winter_simulation/output.jl"); export read_output

###############################################################################
# set parameters

g = 9.81
L = 21.5
height = 5.0
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
ρ_LI = LinearInterpolation(nodes, vcat(1.0 .* ones(59,)))
v_LI = LinearInterpolation(nodes, vcat(1.0 .* ones(59,)))
T_LI = LinearInterpolation(nodes, vcat(1.0 .* ones(59,)))
Ti_LI = LinearInterpolation(nodes, vcat(1.0 .* ones(59,)))
p0 = 1.0

###############################################################################
# Semidiscretization

equations = PassiveHouseEquations1D(;γ, k_i, k_w, k_s, q₀, tᵣ, uᵣ, xa, xb, xc, h=height, L, β, η, Fr², ρₕ₀, T_ref, t_ref, ρ_LI, v_LI, T_LI, Ti_LI, p0)

nodes, t, rho, v, T, Ti, Tu, p0, Tinside = read_output()

#------------------------------------------------------------------------------
### Plot flow quantities ###
@muladd @inline function plot_solution(plot_dir, nodes, time, rho, v, T, Ti, Tu, p0, Tinside, equations)
    plot_dir2 = joinpath(plot_dir, "heatmaps")
    if isdir(plot_dir2) == false
        mkdir(plot_dir2)
    end
    L = length(time)
    t_min = time[1]
    t_max = time[end]
    X = Ref(zeros(L, L))
    Z0 = Ref(zeros(L, L))
    Z1 = Ref(zeros(L, L))
    Z2 = Ref(zeros(L, L))
    Z3 = Ref(zeros(L, L))
    Z4 = Ref(zeros(L, L))
    #-----------------------
    for i = 1:L
        X_now = X[]
        Z0_now = Z0[]
        Z1_now = Z1[]
        Z2_now = Z2[]
        Z3_now = Z3[]
        Z4_now = Z4[]
        #-----------------------
        idx1 = (i-1)*59 + 1
        idx2 = i*59
        #-----------------------
        x_new = range(nodes[1], nodes[end], L)
            rho_new, v_new, T_new, Tu_new = if sum( i .== 2:2:L-1 ) == 1
                  (0.5 * (rho[idx1-59:idx2-59] .+ rho[idx1+59:idx2+59]), 0.5 * (v[idx1-59:idx2-59] .+ v[idx1+59:idx2+59]), 0.5 * (T[idx1-59:idx2-59] .+ T[idx1+59:idx2+59]), 0.5 * (Tu[idx1-59:idx2-59] .+ Tu[idx1+59:idx2+59]))
            else
                  (rho[idx1:idx2], v[idx1:idx2], T[idx1:idx2], Tu[idx1:idx2])
            end
        Ti_new = Ti[idx1:idx2]
        #-----------------------
        rho_LI = LinearInterpolation(nodes, rho_new)
        v_LI = LinearInterpolation(nodes, v_new)
        T_LI = LinearInterpolation(nodes, T_new)
        Tu_LI = LinearInterpolation(nodes, Tu_new)
        Ti_LI = LinearInterpolation(nodes, Ti_new)
        #-----------------------
        X_now[1:L,i] = x_new
        Z0_now[1:L,i] = [rho_LI(x) for x in x_new]
        Z1_now[1:L,i] = [v_LI(x) for x in x_new]
        Z2_now[1:L,i] = [T_LI(x) for x in x_new]
        Z3_now[1:L,i] = [Tu_LI(x) for x in x_new]
        Z4_now[1:L,i] = [Ti_LI(x) for x in x_new]
        #-----------------------
        Z4_now[1,i] = sum([Ti_LI(x) for x in nodes[10:52]])./43
        #-----------------------
        X[] = X_now
        Z0[] = Z0_now
        Z1[] = Z1_now
        Z2[] = Z2_now
        Z3[] = Z3_now
        Z4[] = Z4_now
    end
    #-----------------------
    x = collect(X[])
    xl = 1.04 * equations.xa * equations.L
    xr = 0.96 * equations.xc * equations.L
    y = repeat(range(t_min, t_max, L)', L, 1) 
    #-----------------------
    all_T = hcat(Z2[], Z3[], Z4[])
    T_max = maximum(all_T)
    T_min = minimum(all_T)
    V_max = maximum(abs.(Z1[]))#(maximum(Z1[]) + maximum(abs.(Z1[])) )/2
    V_min = -maximum(abs.(Z1[]))#(minimum(Z1[]) - maximum(abs.(Z1[])) )/2
    #-----------------------
    Plots.gr()
    default(guidefont=font(16), tickfont=font(12), titlefont=font(18), legendfont=font(14))
    Plots.gr_cbar_width[] = 0.05
    #-----------------------
    hot_cold = cgrad([:blue, :red])
    cmap = cgrad([:cyan, :white, :green], rev=true)
    #-----------------------
    L = length(t)
    #-----------------------
    ylabell =  "t (h)"
    #-----------------------
    p_p0 = plot(t[1:2:end-1], p0[1:2:end-1], 
                title = " ", legende = false, size=(700, 550),
                linecolor=:blue, linestyle=:solid, 
                xlabel=ylabell, ylabel="p₀", label = "")
    savefig(p_p0, joinpath(plot_dir, "p0.png"))
    #-----------------------
    p_T_inside = plot(t, Tinside, 
                title = " ", legende = false, size=(700, 550),
                linecolor=:red, linestyle=:solid, 
                xlabel=ylabell, ylabel="T_inside", label = "")
    savefig(p_T_inside, joinpath(plot_dir, "T_inside.png"))
    #-----------------------
    p_rho = surface(x, y, Z0[], 
                    title = " ", legende = false, size=(700, 550),
                    xlabel="x (m)", ylabel=ylabell, zlabel="ρ (kg/m^3)")
    savefig(p_rho, joinpath(plot_dir, "rho.png"))
    plot!(p_rho, xlabel="", ylabel="", zlabel="",
            camera=(0, 90), colorbar_title=" ", yticks=[0, 50, 100, 150], zticks=nothing, colorbar = true)
    savefig(p_rho, joinpath(plot_dir2, "rho.png"))
    #-----------------------
    p_T = surface(x, y, Z2[],
                    title = " ", legende = false, size=(700, 550),
                    c = hot_cold, 
                    xlabel="x (m)", ylabel=ylabell, zlabel="T (°C)");
    savefig(p_T, joinpath(plot_dir, "T.png"))
    plot!(p_T, xlabel="", ylabel="", zlabel="",
            camera=(0, 90), colorbar_title=" ", yticks=[0, 50, 100, 150], zticks=nothing, colorbar = true)
    savefig(p_T, joinpath(plot_dir2, "T.png"))
    #-----------------------
    p_Ti = surface(x, y, Z4[], 
                    title = " ", legende = false, size=(700, 550),
                    c = hot_cold, 
                    xlabel="x (m)", ylabel=ylabell, zlabel="Tᵢ (°C)",
                    xlim = (equations.xa * equations.L, equations.xc * equations.L))           
    savefig(p_Ti, joinpath(plot_dir, "T_i.png"))
    plot!(p_Ti, xlabel="", ylabel="", zlabel="",
            camera=(0, 90), colorbar_title=" ", yticks=[0, 50, 100, 150], zticks=nothing, colorbar = true)
    savefig(p_Ti, joinpath(plot_dir2, "T_i.png"))
    #-----------------------
    p_Tu = surface(x, y, Z3[], 
                    title = " ", legende = false, size=(700, 550),
                    c = hot_cold, 
                    xlabel="x (m)", ylabel=ylabell, zlabel="Tᵤ (°C)")
        surface!(x .+ 100, y, fill(T_max, length(time),length(time)), c = hot_cold, label = "")   
        surface!(x .+ 100, y, fill(T_min, length(time),length(time)), c = hot_cold, label = "", xlim=(0.0, equations.L))   
    savefig(p_Tu, joinpath(plot_dir, "T_u.png"))
    plot!(p_Tu, xlabel="", ylabel="", zlabel="",
            camera=(0, 90), colorbar_title=" ", yticks=[0, 50, 100, 150], zticks=nothing, colorbar = true)  
    savefig(p_Tu, joinpath(plot_dir2, "T_u.png"))
    #-----------------------
    surface(x, y, Z1[], c = cmap, xlabel="x (m)", ylabel="       t [h]", zlabel="u (cm/s)", title="", colorbar = false)
    cmap = cgrad([:cyan, :white, :green], rev=true)
    #cmap = cgrad([:cyan, :lightgray, :green], rev=true)
    p_u = surface(x, y, Z1[], 
                    title = " ", legende = false, size=(700, 550),
                    c = cmap, 
                    xlabel="x (m)", ylabel=ylabell, zlabel="u (cm/s)")
        p_u = surface!(p_u, x .+ 100, y, fill(V_max, length(time),length(time)), c = cmap, label = "")
        p_u = surface!(p_u, x .+ 100, y, fill(V_min, length(time),length(time)), c = cmap, label = "", xlim=(0.0, equations.L))   
    savefig(p_u, joinpath(plot_dir, "u.png"))
    p_u = plot!(p_u, xlabel="", ylabel="", zlabel="",
            camera=(0, 90), colorbar_title=" ", yticks=[0, 50, 100, 150], zticks=nothing, colorbar = true) 
    savefig(p_u, joinpath(plot_dir2, "u.png"))
    #-----------------------
    return nothing
end
#------------------------------------------------------------------------------
### Average temperature function ###
@muladd @inline function avg_temp(t, T_avg)
      tt = mod.(t, 24)
      for i = 1:length(t)
            if tt[i] ≤ 8.0
                  tt[i] = 0.0
            elseif tt[i] ≤ 20.0
                  tt[i] = 1.0
            else
                  tt[i] = 0.0
            end
      end
      daytime = if sum(tt) == 0.0
            0.0
      else
            sum(tt .* T_avg) ./ sum(tt)
      end
      nighttime = if sum(- (tt .- 1)) == 0.0
            0.0
      else
            sum(- (tt .- 1) .* T_avg) ./ sum(- (tt .- 1))
      end
      return daytime, nighttime, tt
end
#------------------------------------------------------------------------------
### Plot profiles ###
@muladd @inline function plot_profiles(output_dir, nodes, t, equations)
    plot_dir4 = joinpath(plot_dir, "profiles")
    if isdir(plot_dir4) == false
        mkdir(plot_dir4)
    end
    #-----------------------
    Plots.gr()
    default(guidefont=font(16), tickfont=font(12), titlefont=font(18), legendfont=font(14))
    Plots.gr_cbar_width[] = 0.05
    #-----------------------
    p_A = plot(nodes,  0.05 .* [A(x, equations) for x in range(0,1,length(nodes))], 
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:blue, linestyle=:solid, 
                        xlabel="x (m)", ylabel="A (m^2)", label = "")
    savefig(p_A, joinpath(plot_dir4, "A.png"))
    #-----------------------
    p_Ax = plot(nodes, [A_x(x, equations) for x in range(0,1,length(nodes))], 
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:blue, linestyle=:dashdot,
                        xlabel="x (m)", ylabel="A_x", label = "")
    savefig(p_Ax, joinpath(plot_dir4, "A_x.png"))
    #-----------------------
    p_h = plot(nodes, [h(x, equations) for x in range(0,1,length(nodes))],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:black, linestyle=:solid,
                        xlabel="x (m)", ylabel="h (m)", label = "")
    savefig(p_h, joinpath(plot_dir4, "h.png"))
    #-----------------------
    p_hx = plot(nodes, [h_x(x, equations) for x in range(0,1,length(nodes))],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:black, linestyle=:dashdot,
                        xlabel="x (m)", ylabel="h_x", label = "")
    savefig(p_hx, joinpath(plot_dir4, "hx.png"))
    #----------------------- 
    p_T_air = plot(range(0.0, 150 .* 86400 ./ equations.tᵣ, 150*12) ./ (86400 ./ equations.tᵣ), T_air.(range(0.0, 150 .* 86400 ./ equations.tᵣ, 150*12), equations) .- 273.15,
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:red, linestyle=:dashdotdot, 
                        xlabel="t (days)", ylabel = "T_air (°C)", label = "")
    savefig(p_T_air, joinpath(plot_dir4, "T_air.png"))
    #-----------------------
    p_T_air = plot(range(0.0, 24, 200), T_air.(range(0.0, 86400 ./ equations.tᵣ, 200), equations) .- 273.15,
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:red, linestyle=:dashdotdot, 
                        xlabel="t (h)", ylabel = "T_air (°C)", label = "")
    savefig(p_T_air, joinpath(plot_dir4, "T_air_day1.png"))
    #-----------------------
    p_T_air = plot(range(0.0, 24, 200), T_air.(range(74 * 86400 ./ equations.tᵣ, 75 * 86400 ./ equations.tᵣ, 200), equations) .- 273.15,
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:red, linestyle=:dashdotdot, 
                        xlabel="t (h)", ylabel = "T_air (°C)", label = "")
    savefig(p_T_air, joinpath(plot_dir4, "T_air_day75.png"))
    #-----------------------
    p_T_soil = plot(range(0.0, 150 .* 86400 ./ equations.tᵣ, 20) ./ (86400 ./ equations.tᵣ), T_soil.(range(0.0, 150 .* 86400 ./ equations.tᵣ, 20), equations) .- 273.15,
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:blue, linestyle=:dashdot, 
                        xlabel="t (days)", ylabel = "T_soil (°C)", label = "")
    savefig(p_T_soil, joinpath(plot_dir4, "T_soil.png"))
    #-----------------------
    p_q_w = plot(range(0.0, 150 .* 86400 ./ equations.tᵣ, 150*12) ./ (86400 ./ equations.tᵣ), equations.q₀ * [Q_s(i, 0.5*(equations.xb+equations.xa), equations) for i in range(0.0, 150 .* 86400 ./ equations.tᵣ, 150*12)],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:green, linestyle=:solid,
                        xlabel="t (days)", ylabel="Irradiance (W/m^2)", label = "")
    savefig(p_q_w, joinpath(plot_dir4, "q_window.png"))
    #-----------------------
    p_q_w = plot(range(0.0, 24, 200), equations.q₀ * [Q_s(i, 0.5*(equations.xb+equations.xa), equations) for i in range(0.0, 86400 ./ equations.tᵣ, 200)],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:green, linestyle=:solid,
                        xlabel="t (h)", ylabel="Irradiance (W/m^2)", label = "")
    savefig(p_q_w, joinpath(plot_dir4, "q_window_day1.png"))
    #-----------------------
    p_q_w = plot(range(0.0, 24, 200), equations.q₀ * [Q_s(i, 0.5*(equations.xb+equations.xa), equations) for i in range(74 * 86400 ./ equations.tᵣ, 75 * 86400 ./ equations.tᵣ, 200)],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:green, linestyle=:solid,
                        xlabel="t (h)", ylabel="Irradiance (W/m^2)", label = "")
    savefig(p_q_w, joinpath(plot_dir4, "q_window_day75.png"))
    #-----------------------
    p_alpha = plot(nodes, 1.1 .* [I_w(x, equations) for x in range(0,1,length(nodes))],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:black, linestyle=:solid,
                        xlabel="x (m)", ylabel="α_w", label = "")
    savefig(p_alpha, joinpath(plot_dir4, "α_w.png"))
    #-----------------------
    return nothing
end

#------------------------------------------------------------------------------
### Create Plots ###
plot_solution(plot_dir, nodes, t, rho, v, T, Ti, Tu, p0, Tinside, equations)
plot_profiles(plot_dir, nodes, t, equations)
#------------------------------------------------------------------------------