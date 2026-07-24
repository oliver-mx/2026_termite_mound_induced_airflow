# Creates plots of the `passive_house_simulation.jl` script
# 

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, OrdinaryDiffEqLowStorageRK, Interpolations, QuadGK, FastGaussQuadrature, Plots
using Trixi: AbstractEquations, @muladd
import Interpolations: Line
import Trixi: flux_ranocha, ln_mean, inv_ln_mean, flux, varnames, cons2cons, cons2prim, prim2cons, cons2entropy, max_abs_speeds

###############################################################################
# Output
out_dir = joinpath(@__DIR__, "../../out")
output_dir = joinpath(out_dir, "passive_house")
if isdir(output_dir) == false
    error("Run passive_house_simulation.jl first!")
end
plot_dir = joinpath(output_dir, "plots")
if isdir(plot_dir) == false
    mkdir(plot_dir)
end

#------------------------------------------------------------------------------
include("../equations/passive_house_1d.jl")
include("../callbacks/update_velocity_passive_house.jl")
include("../../out/passive_house/output.jl"); export read_output

###############################################################################
# Semidiscretization

equations = PassiveHouseEquations1D()

nodes, t, rho, v, T, Ti, Tu, p0, Tinside = read_output()

#------------------------------------------------------------------------------
### Plot flow quantities ###
@muladd @inline function plot_solution(plot_dir, nodes, time, rho, v, T, Ti, Tu, p0, Tinside, equations)
    #
    println("T_inside: $(round(Tinside[end]-Tinside[1], digits=3))°C")
    #
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
        rho_new = rho[idx1:idx2]
        v_new = v[idx1:idx2]
        T_new = T[idx1:idx2]
        Tu_new = Tu[idx1:idx2]
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
    p_p0 = plot(t, p0, 
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
            camera=(0, 90), colorbar_title=" ", yticks=[0,4,8,12,16,20,24], zticks=nothing, colorbar = true)
    savefig(p_rho, joinpath(plot_dir2, "rho.png"))
    #-----------------------
    p_T = surface(x, y, Z2[],
                    title = " ", legende = false, size=(700, 550),
                    c = hot_cold, 
                    xlabel="x (m)", ylabel=ylabell, zlabel="T (°C)");
    savefig(p_T, joinpath(plot_dir, "T.png"))
    plot!(p_T, xlabel="", ylabel="", zlabel="",
            camera=(0, 90), colorbar_title=" ", yticks=[0,4,8,12,16,20,24], zticks=nothing, colorbar = true)
    savefig(p_T, joinpath(plot_dir2, "T.png"))
    #-----------------------
    p_Ti = surface(x, y, Z4[], 
                    title = " ", legende = false, size=(700, 550),
                    c = hot_cold, 
                    xlabel="x (m)", ylabel=ylabell, zlabel="Tᵢ (°C)",
                    xlim = (equations.xa * equations.L, equations.xc * equations.L))           
    savefig(p_Ti, joinpath(plot_dir, "T_i.png"))
    plot!(p_Ti, xlabel="", ylabel="", zlabel="",
            camera=(0, 90), colorbar_title=" ", yticks=[0,4,8,12,16,20,24], zticks=nothing, colorbar = true)
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
            camera=(0, 90), colorbar_title=" ", yticks=[0,4,8,12,16,20,24], zticks=nothing, colorbar = true)  
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
            camera=(0, 90), colorbar_title=" ", yticks=[0,4,8,12,16,20,24], zticks=nothing, colorbar = true) 
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
    p_T_air = plot(t, T_air.(t, equations) .- 273.15,
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:red, linestyle=:dashdotdot, 
                        xlabel="t (h)", ylabel = "T_air (°C)", label = "")
    savefig(p_T_air, joinpath(plot_dir4, "T_air.png"))
    #-----------------------
    p_T_soil = plot(t, T_soil.(t, equations) .- 273.15,
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:blue, linestyle=:dashdot, 
                        xlabel="t (h)", ylabel = "T_soil (°C)", label = "")
    savefig(p_T_soil, joinpath(plot_dir4, "T_soil.png"))
    #-----------------------
    p_alpha = plot(nodes, 1.1 .* [I_w(x, equations) for x in range(0,1,length(nodes))],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:black, linestyle=:solid,
                        xlabel="x (m)", ylabel="α_w", label = "")
    savefig(p_alpha, joinpath(plot_dir4, "α_w.png"))
    #-----------------------
    I = LinearInterpolation([0.0, equations.xa - 1e-8, equations.xa, equations.xb, equations.xb + 1e-8, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    p_q_0 = plot(space2unscaled.(range(0,1,length(nodes)), equations), equations.q₀ .* [I(x) .* Q_s(0.0, x, equations) for x in range(0,1,length(nodes))],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:green, linestyle=:solid,
                        xlabel="x (m)", ylabel="Irradiance (W/m^2)", label = "")
    savefig(p_q_0, joinpath(plot_dir4, "q_midnight.png"))
    #----------------------- 
    p_q_12 = plot(space2unscaled.(range(0,1,length(nodes)), equations), equations.q₀ .* [I(x) .* Q_s(0.5 * 86400 ./ equations.tᵣ, x, equations) for x in range(0,1,length(nodes))],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:green, linestyle=:solid,
                        xlabel="x (m)", ylabel="Irradiance (W/m^2)", label = "")
    savefig(p_q_12, joinpath(plot_dir4, "q_noon.png"))
    #-----------------------
    z=equations.q₀
    xs=[0, 7, 10.5, 12, 13, 15, 16.5, 18.5, 24];
    ys=([-10, -10, 0.3*z, 0.6 * z, z, z, 0, -10, -10]);
    LI=LinearInterpolation(xs, ys);
    p_q_w = plot(range(0,24,length(nodes)), [LI(x) for x in range(0,24,length(nodes))],
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:black, linestyle=:solid,
                        xlabel="t (h)", ylabel="Irradiance (W/m^2)", label = "")
    savefig(p_q_w, joinpath(plot_dir4, "q_window.png"))
    #-----------------------
    return nothing
end

#------------------------------------------------------------------------------
### Create Plots ###
plot_solution(plot_dir, nodes, t, rho, v, T, Ti, Tu, p0, Tinside, equations)
plot_profiles(plot_dir, nodes, t, equations)
#------------------------------------------------------------------------------