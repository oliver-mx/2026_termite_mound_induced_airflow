# Creates plots of the `optimal_mound_simulation.jl` script
# 

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, OrdinaryDiffEqLowStorageRK, Interpolations, QuadGK, FastGaussQuadrature, Plots
using Trixi: AbstractEquations, @muladd
import Interpolations: Line
import Trixi: flux_ranocha, ln_mean, inv_ln_mean, flux, varnames, cons2cons, cons2prim, prim2cons, cons2entropy, max_abs_speeds

###############################################################################
# Output

out_dir = joinpath(@__DIR__, "../../out")
output_dir = joinpath(out_dir, "optimal_mound")
if isdir(output_dir) == false
    error("Run optimal_mound_simulation.jl first!")
end
plot_dir = joinpath(output_dir, "plots")
if isdir(plot_dir) == false
    mkdir(plot_dir)
end

#------------------------------------------------------------------------------
include("../equations/termite_mound_1d.jl")
include("../callbacks/update_velocity_optimal.jl")
include("../../out/optimal_mound/output.jl"); export read_output

###############################################################################
# set parameters

r=0.7448388482537119
height=2.41676760522155
#
g = 9.81
L = height + 0.3 + sqrt(0.09 + r*r) + sqrt(r*r + height*height)
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
xb = (sqrt(0.09 + r*r) + sqrt(r*r + height*height)) / L
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
mass = (1742.448 * (1/3 * pi * r * r * height) ) ./ 8 
Li = (xᵣ*(xc-xa))
κ = 0.184 
k_i = tᵣ * κ / (c_v * mass/Li)
T_ref=297.76980029566033
t_ref=0.0033582989242263127
T0 = 27.0
v0 = -2.219
Ti_LI=LinearInterpolation([0.0, xb, 1.0],[27.6, 28.8, 27.5])

###############################################################################
# Semidiscretization

equations = TermiteMoundEquations1D(;γ, k_i, k_w, tᵣ, uᵣ, xa, xb, xc, r, h=height, L, β, η, Fr², ρₕ₀, T_ref, t_ref, T0, v0, Ti_LI)

nodes, t, rho, v, T, Ti, Tu, p0, L2_error, Vel_switches = read_output()

#------------------------------------------------------------------------------
### Plot flow quantities ###

@muladd @inline function plot_solution(plot_dir, nodes, time, rho, v, T, Ti, Tu, p0, equations)
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
    V_max = maximum(abs.(Z1[]))
    V_min = -maximum(abs.(Z1[]))
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
    try
    p_rho = surface(x, y, Z0[], 
                    title = " ", legende = false, size=(700, 550),
                    xlabel="x (m)", ylabel=ylabell, zlabel="ρ (kg/m^3)")
    savefig(p_rho, joinpath(plot_dir, "rho.png"))
    plot!(p_rho, xlabel="", ylabel="", zlabel="",
            camera=(0, 90), colorbar_title=" ", yticks=[0,4,8,12,16,20,24], zticks=nothing, colorbar = true)
    savefig(p_rho, joinpath(plot_dir2, "rho.png"))
    catch
        println("Warning: Failed to create `rho.png`!")
    end
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
### Comparison with experiment ###

@muladd @inline function plot_experiment(output_dir, nodes, time, rho, v, T, Ti, Tu, equations)
    plot_dir3 = joinpath(plot_dir, "experiment")
    if isdir(plot_dir3) == false
        mkdir(plot_dir3)
    end
    #-----------------------
    Plots.gr()
    default(guidefont=font(16), tickfont=font(12), titlefont=font(18), legendfont=font(14))
    Plots.gr_cbar_width[] = 0.05
    #-----------------------
    L = length(time)
    x = range(0.0, time[end], L)
    #-----------------------
    Velocity = ([v[(i-1)*59 + 1] for i = 1:L] .+ [v[(i-1)*59 + 56] for i = 1:L] .+ [v[(i-1)*59 + 57] for i = 1:L] .+ [v[(i-1)*59 + 58] for i = 1:L] .+ [v[(i-1)*59 + 59] for i = 1:L]) ./ 5.0
    Nest = ([T[(i-1)*59 + 1] for i = 1:L] .+ [T[(i-1)*59 + 56] for i = 1:L] .+ [T[(i-1)*59 + 57] for i = 1:L] .+ [T[(i-1)*59 + 58] for i = 1:L] .+ [T[(i-1)*59 + 59] for i = 1:L]) ./ 5.0
    Chimney_mid = [T[(i-1)*59 + 42] for i = 1:L]
    #-----------------------
    dead_vel_x_data = [0.0, 0.0850202429149794, 0.206477732793522, 0.327935222672066, 0.558704453441296, 0.716599190283402, 0.874493927125507, 0.99595141700405, 1.25101214574899, 1.44534412955466, 1.57894736842105, 1.8582995951417, 2.04048582995951, 2.35627530364373, 2.55060728744939, 2.81781376518219, 2.92712550607287, 3.17004048582996, 3.30364372469636, 3.57085020242915, 3.81376518218624, 3.99595141700405, 4.12955465587045, 4.37246963562753, 4.53036437246964, 4.84615384615385, 5.06477732793523, 5.33198380566802, 5.51417004048583, 5.79352226720648, 6.02429149797571, 6.21862348178138, 6.41295546558705, 6.65587044534413, 6.91093117408907, 7.06882591093118, 7.26315789473684, 7.4089068825911, 7.60323886639676, 7.72469635627531, 7.87044534412956, 7.9919028340081, 8.02834008097166, 8.07692307692308, 8.17408906882591, 8.25910931174089, 8.29554655870446, 8.40485829959514, 8.42914979757085, 8.52631578947369, 8.82995951417004, 9.06072874493927, 9.34008097165992, 9.59514170040486, 9.91093117408907, 10.1174089068826, 10.3117408906883, 10.5303643724696, 10.7611336032389, 10.9919028340081, 11.1255060728745, 11.331983805668, 11.4291497975709, 11.4655870445344, 11.5506072874494, 11.5991902834008, 11.7813765182186, 11.8663967611336, 11.9514170040486, 12.0242914979757, 12.085020242915, 12.1457489878543, 12.2672064777328, 12.2672064777329, 12.3279352226721, 12.4372469635628, 12.582995951417, 12.6315789473684, 12.8744939271255, 13.0931174089069, 13.2874493927126, 13.5060728744939, 13.6882591093117, 13.9433198380567, 14.1376518218624, 14.3441295546559, 14.5991902834008, 14.7935222672065, 15.0121457489879, 15.2914979757085, 15.497975708502, 15.7165991902834, 15.8866396761134, 16.0931174089069, 16.2995951417004, 16.6275303643725, 16.8218623481781, 17.0283400809717, 17.1376518218624, 17.2834008097166, 17.417004048583, 17.587044534413, 17.7692307692308, 17.8542510121457, 17.9514170040486, 18.0, 18.085020242915, 18.1578947368421, 18.2672064777328, 18.3157894736842, 18.4372469635628, 18.5465587044534, 18.7165991902834, 18.7894736842105, 18.8987854251012, 19.0080971659919, 19.1295546558705, 19.2024291497976, 19.3603238866397, 19.4817813765182, 19.6761133603239, 19.919028340081, 20.1497975708502, 20.3805668016194, 20.4777327935223, 20.6477732793522, 20.7813765182186, 20.8906882591093, 21.0242914979757, 21.2064777327935, 21.3036437246964, 21.6072874493927, 21.7287449392713, 21.8987854251012, 22.0323886639676, 22.1781376518219, 22.336032388664, 22.5303643724696, 22.6761133603239, 22.8582995951417, 23.0283400809717, 23.3076923076923, 23.4898785425101, 23.6599190283401, 23.7935222672065, 23.9271255060729, 24.0]
    dead_vel_y_data = [-1.85749385749385, -1.85749385749386, -1.88697788697787, -1.88697788697788, -1.9017199017199, -1.88697788697788, -1.88697788697789, -1.97542997542995, -1.97542997542996, -1.97542997542997, -1.97542997542998, -2.06388206388206, -2.07862407862407, -2.07862407862408, -2.15233415233413, -2.15233415233414, -2.15233415233415, -2.1965601965602, -2.21130221130221, -2.28501228501226, -2.28501228501227, -2.28501228501228, -2.37346437346438, -2.32923832923833, -2.37346437346435, -2.37346437346436, -2.37346437346437, -2.38820638820639, -2.41769041769042, -2.32923832923833, -2.25552825552825, -2.1965601965602, -2.12285012285012, -2.06388206388206, -1.94594594594595, -1.7985257985258, -1.63636363636364, -1.48894348894349, -1.26781326781327, -1.04668304668305, -0.796068796068795, -0.545454545454545, -0.309582309582309, -0.206388206388206, 0.0442260442260451, 0.457002457002457, 0.604422604422605, 0.86977886977887, 1.01719901719902, 1.2972972972973, 1.48894348894349, 1.54791154791154, 1.56265356265356, 1.59213759213759, 1.51842751842752, 1.53316953316953, 1.54791154791155, 1.63636363636364, 1.75429975429976, 1.88697788697788, 1.88697788697789, 1.9017199017199, 1.94594594594595, 1.96068796068796, 2.32923832923833, 2.01965601965602, 2.4029484029484, 2.24078624078624, 2.50614250614251, 2.41769041769042, 2.31449631449632, 2.00491400491401, 2.15233415233415, 1.97542997542997, 1.97542997542998, 1.93120393120393, 1.9017199017197, 1.9017199017198, 1.9017199017199, 1.81326781326781, 1.78378378378377, 1.78378378378378, 1.78378378378379, 1.75429975429976, 1.73955773955774, 1.71007371007371, 1.68058968058968, 1.65110565110565, 1.63636363636364, 1.68058968058967, 1.62162162162162, 1.56265356265356, 1.54791154791155, 1.53316953316953, 1.48894348894349, 1.45945945945946, 1.44471744471744, 1.4004914004914, 1.35626535626536, 1.19410319410319, 1.10565110565111, 0.943488943488944, 0.796068796068797, 0.61916461916462, 0.412776412776413, 0.176904176904177, 0.0147420147420156, -0.176904176904176, -0.353808353808353, -0.589680589680589, -0.78132678132678, -0.928746928746928, -1.14987714987715, -1.26781326781327, -1.4004914004914, -1.5036855036855, -1.62162162162162, -1.66584766584767, -1.7985257985258, -1.84275184275184, -1.94594594594595, -2.01965601965602, -2.07862407862408, -2.12285012285012, -2.16707616707616, -2.16707616707617, -2.10810810810811, -2.07862407862408, -2.03439803439803, -1.99017199017199, -1.99017199017199, -1.93120393120393, -1.93120393120392, -1.93120393120393, -1.94594594594594, -1.94594594594595, -1.97542997542998, -1.97542997542997, -1.97542997542998, -1.99017199017199, -2.01965601965602, -2.10810810810812, -2.10810810810811, -2.12285012285012, -2.10810810810810, -2.10810810810811, -2.10810810810812]
    alive_vel_x_data = [1.61866125760649, 1.4604462474645, 2.06896551724138, 3.15212981744422, 3.57809330628803, 3.84584178498986, 4.80730223123732, 4.96551724137931, 5.63488843813387, 5.82961460446248, 6.24340770791075, 6.02434077079107, 7.10750507099391, 7.2657200811359, 7.3630831643002, 7.56997971602434, 7.87423935091278, 7.947261663286, 7.86206896551724, 8.08113590263692, 8.21501014198783, 8.4340770791075, 8.51926977687627, 9.17647058823529, 9.26166328600406, 9.51724137931035, 9.62677484787018, 9.87018255578093, 10.0649087221095, 10.4300202839757, 10.2231237322515, 10.7951318458418, 10.709939148073, 11.0628803245436, 11.2089249492901, 12.3286004056795, 12.4868154158215, 12.8275862068966, 13.3022312373225, 13.7525354969574, 13.9107505070994, 13.8985801217039, 14.0567951318458, 14.3002028397566, 15.0547667342799, 15.237322515213, 16.2596348884381, 16.7707910750507, 16.5760649087221, 16.6369168356998, 16.369168356998, 17.3549695740365, 17.4523326572008, 17.7565922920892, 17.8661257606491, 18.0730223123732, 18.1095334685598, 18.3651115618661, 18.5598377281947, 19.5212981744422, 19.3752535496957, 19.7403651115619, 20.0202839756592, 20.105476673428, 20.5679513184584, 20.9939148073022, 20.5922920892495, 21.5172413793103, 21.395537525355, 21.2494929006085, 21.6754563894523, 21.9675456389452, 22.1135902636917, 22.5760649087221, 22.8803245436105]
    alive_vel_y_data = [1.22358722358722, -3.56756756756757, 0.0884520884520876, -4.93857493857494, -4.96805896805897, -5.01228501228501, -5.05651105651106, -2.32923832923833, -4.92383292383292, -4.96805896805897, -4.98280098280098, -1.62162162162162, -4.96805896805897, -4.96805896805897, 0.206388206388207, 0.368550368550369, -1.57739557739558, -1.48894348894349, -4.96805896805897, -4.57002457002457, -2.16707616707617, -4.52579852579853, -3.22850122850123, 1.23832923832924, 1.73955773955774, 1.75429975429975, 1.12039312039312, 1.35626535626536, 2.15233415233415, 3.87714987714988, 0.825552825552825, 0.383292383292384, 1.20884520884521, 1.51842751842752, 2.49140049140049, 1.6953316953317, 1.44471744471744, 0.574938574938575, 3.14004914004914, 2.88943488943489, 1.68058968058968, 1.26781326781327, 0.457002457002457, 2.65356265356265, 1.59213759213759, 1.32678132678133, -1.94594594594595, -1.35626535626536, -0.398034398034399, -0.103194103194103, 0.147420147420148, 0.796068796068796, 0.28009828009828, 1.14987714987715, 2.56511056511057, -0.117936117936119, 0.30958230958231, 0.560196560196561, 0.855036855036855, -4.18673218673219, -5.01228501228501, -4.98280098280098, -5.02702702702703, -5.01228501228501, -5.01228501228501, -4.98280098280098, -1.44471744471744, -0.0737100737100738, -2.4029484029484, -2.75675675675676, -2.1965601965602, -3.56756756756757, -4.96805896805897, -4.93857493857494, -3.33169533169533]
    #-----------------------
    p_vel = plot(dead_vel_x_data, dead_vel_y_data, 
        title = " ", size=(700, 550), legend=:bottomright,
        #xlabel="t (h)", ylabel="Velocity (m/s)", 
        linecolor=:green, label = "Dead mounds")
        scatter!(alive_vel_x_data, alive_vel_y_data, label = "Alive mounds", markercolor=:green, markersize=:4.0, marker=:xcross)  
        plot!(x, Velocity, label = "u(t,0)",xticks = [0,4,8,12,16,20,24], linecolor=:black)
    savefig(p_vel, joinpath(plot_dir3, "Air_velocity.png"))
    #-----------------------
    p_Tnest = plot(t, T_air.(t, equations) .- 273.15,
                title = " ", size=(700, 550), legend=:bottomright,
                linecolor=:red, linestyle=:dashdotdot, 
                #xlabel="t (h)", ylabel="Temperature (°C)",
                label = "T        ",#"T_air",
                xticks = [0,4,8,12,16,20,24], ylim=(19.0, 34.0), pos = :top)
              plot!(t, T_soil.(t, equations) .- 273.15, 
                linecolor=:blue, linestyle=:dashdot, 
                label = "T        ")#"T_soil")
              plot!(x, T_nest.(x, equations) .- 273.15, 
                linecolor=:green, linestyle=:solid, 
                label = "Data")
              plot!(x, Nest,
                linecolor=:black, linestyle=:dash,label = "T(t,0)")
    savefig(p_Tnest, joinpath(plot_dir3, "Nest_temp.png"))
    #-----------------------
    return nothing
end

#------------------------------------------------------------------------------
### Measured Nest Temperature ###
@inline function T_nest(t_var, equations::TermiteMoundEquations1D)
    a0 =       300.1;    a1 =     0.05284;    a10 =   -0.008549;    a2 =     0.06245;    a3 =    -0.01747;    a4 =    -0.04914;    a5 =    -0.01773;    a6 =    -0.01709;    a7 =   -0.007458;    a8 =    -0.01166;    a9 =    0.007433;
    return Fourir(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, t_var, equations::TermiteMoundEquations1D) #* 0.0033582989242263127
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
    p_T_air = plot(t, T_soil.(t, equations) .- 273.15,
                        title = " ", legende = false, size=(700, 550),
                        linecolor=:blue, linestyle=:dashdot, 
                        xlabel="t (h)", ylabel = "T_soil (°C)", label = "")
    savefig(p_T_air, joinpath(plot_dir4, "T_soil.png"))
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
plot_solution(plot_dir, nodes, t, rho, v, T, Ti, Tu, p0, equations)
plot_profiles(plot_dir, nodes, t, equations)
plot_experiment(plot_dir, nodes, t, rho, v, T, Ti, Tu, equations)
#------------------------------------------------------------------------------