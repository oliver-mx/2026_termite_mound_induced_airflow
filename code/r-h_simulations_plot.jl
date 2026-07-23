# Plots the results from `r-h_simulations.jl`.
# 
# WARNING: Only run this file after all simulations are done.
#
# All function environments in all `r-h_simulation` output files will be closed when this skript is executed.
# If a simulations is still ongoing it will add code outside of the read_output function, causing errors.   
#

# using TermiteMoundInducedAirflowTrixi.jl
using Trixi, Plots, Interpolations, QuadGK
using Trixi: AbstractEquations, @muladd
import Interpolations: Line
import Trixi: flux_ranocha, ln_mean, inv_ln_mean, flux, varnames, cons2cons, cons2prim, prim2cons, cons2entropy, max_abs_speeds

###############################################################################
# Output

out_dir = joinpath(@__DIR__, "out")
output_dir = joinpath(out_dir, "r-h_simulations")
if isdir(output_dir) == false
    error("Run r-h_experiment.jl first!")
end
Number_of_Instances = Ref(1)

for name in readdir(output_dir)
    if name == "plots"
        continue
    end
    path = joinpath(output_dir, name)
    NoI = try 
            parse(Int, name[8:9])
        catch
            parse(Int, name[8])
        end
    Number_of_Instances[] = maximum([NoI, Number_of_Instances[]])
    if isfile(path)
        s = read(output_dir * "//" * name, String)
        s_end = last(s, min(84, lastindex(s)))
        if s_end == "end\n#------------------------------------------------------------------------------\n"
            nothing
        else
            open(joinpath(output_dir, name), "a") do file
                    write(file, "\n    return r, h, L2_error, Vel_switches\nend\n#------------------------------------------------------------------------------\n")
                    flush(file)
            end 
        end 
    end
end

plot_dir = joinpath(output_dir, "plots")
if isdir(plot_dir) == false
    mkdir(plot_dir)
end

#------------------------------------------------------------------------------
include("src/equations/termite_mound_1d.jl")
include("src/callbacks/update_velocity_r-h.jl")

for i in 1:Number_of_Instances[]
    include("out\\r-h_simulations/output_$i.jl")
end

equations = TermiteMoundEquations1D()
radius = Ref(zeros(961,))
height = Ref(zeros(961,))
L2 = Ref(zeros(961,))
V = Ref(zeros(961,))
Pos = Ref(zeros(961,))
idx = Ref(1)

for i in 1:Number_of_Instances[]
    #
    radius_r = radius[]; height_r = height[]; L2_r = L2[]; V_r = V[]; Pos_r = Pos[];
    #
    radius_i, height_i, L2_i, V_i = if i==1
                                        read_output_1()
                                    elseif i==2
                                        read_output_2()
                                    elseif i==3
                                        read_output_3()
                                    elseif i==4
                                        read_output_4()
                                    elseif i==5
                                        read_output_5()
                                    elseif i==6
                                        read_output_6()
                                    elseif i==7
                                        read_output_7()
                                    elseif i==8
                                        read_output_8()
                                    elseif i==9
                                        read_output_9()
                                    elseif i==10
                                        read_output_10()
                                    elseif i==11
                                        read_output_11()
                                    elseif i==12
                                        read_output_12()
                                    elseif i==13
                                        read_output_13()
                                    elseif i==14
                                        read_output_14()
                                    elseif i==15
                                        read_output_15()
                                    elseif i==16
                                        read_output_16()
                                    end
    #
    L_i = length(radius_i) - 1                          
    radius_r[idx[]:idx[]+L_i-1] = radius_i[2:end]
    height_r[idx[]:idx[]+L_i-1] = height_i[2:end]
    L2_r[idx[]:idx[]+L_i-1] = L2_i[2:end]
    V_r[idx[]:idx[]+L_i-1] = V_i[2:end]
    Pos_r[idx[]:idx[]+L_i-1] = i .* ones(L_i,)
    idx_r = idx[] + L_i
    #
    radius[] = radius_r; height[] = height_r; L2[] = L2_r; V[] = V_r; Pos[] = Pos_r; idx[] = idx_r
end

@muladd @inline function plot_solution(plot_dir, radius, height, L2, V, Pos, equations)

    idx = findall(V .== 0)
    
    #height = repeat(range(0.5, 3.0, length=31), inner=31)
    #radius = repeat(range(0.1, 1.5, length=31), outer=31)
    #L2 = rand(31*31,)
    #V = round.( rand(31*31,), digits=0)

    positive = findall(L2 .> 0)
    minL2 = positive[argmin(L2[positive])]
    println("Best simulated setup: \nr = $(radius[minL2]), h = $(height[minL2])")
    println("(ϵ, ζ) = ($(L2[minL2]), $(V[minL2]))")

    Plots.gr()
    default(guidefont=font(16), tickfont=font(12), titlefont=font(18), legendfont=font(14))
    Plots.gr_cbar_width[] = 0.05

    p_L2 = surface(radius, height, L2, 
                    title=" ", legend = :top, size=(700, 550),
                    colorbar_title=" ", 
                    ylim = (0.5, 2.5), xlim = (0.3, 1.5), camera=(0, 90.0),
                    label = "", xlabel="", ylabel="", zlabel="", zticks=nothing);
    savefig(p_L2, joinpath(plot_dir, "L2.png"))

    p_V = surface(radius, height, V,
                    title="", legend = :top, size=(700, 550),
                    colorbar_title=" ", 
                    ylim = (0.5, 2.5), xlim = (0.3, 1.5), camera=(0, 90.0),
                    label = "", xlabel="", ylabel="", zlabel="", zticks=nothing);

    radius2 = range(0.3, 1.5, length=31)
    height2 = range(0.5, 2.5, length=31)
    L2_c = reshape(L2, 31, 31)'
    p_L2_c = contourf(radius2, height2, L2_c,
        title="",
        size=(700, 550),
        #levels = 50,
        colorbar_title=" ",
        ylim=(0.5, 2.5), xlim=(0.3, 1.5),
        xlabel="", ylabel="", dpi=300
        #legend=false
    )
    #savefig(p_L2_c, joinpath(plot_dir, "L2_.png"))

    ### data from Bera et al
    hb = [1.7780, 1.2192, 2.4130, 2.0320, 1.5240, 1.6510, 0.9398] # 0.3322, 2.0574, 1.0160,
    rb = [2.8448, 1.3208, 2.7940, 1.7780, 1.6764, 2.0828, 1.1176] ./ 2 # 1.1684, 3.1496, 3.0480,
    p_V = scatter!(p_V, rb, hb, ones(length(hb)), marker=:circle, markersize=6, color=:green, label="Data")
    p_V = scatter(p_V, [0.6], [2.0], [1.0], marker=:diamond, markersize=6, color=:red, label="Reference")
    p_Pos = surface(radius, height, Pos, grid = true,
                    title="", legend = :top, size=(700, 550),
                    colorbar_title=" ", colormap = :viridis,
                    ylim = (0.5, 2.5), xlim = (0.3, 1.5), camera=(0, 90.0),
                    label = "", xlabel="", ylabel="", zlabel="", zticks=nothing);        
    savefig(p_Pos, joinpath(plot_dir, "i.png"))
    savefig(p_V, joinpath(plot_dir, "V.png"))
    return nothing
end

#------------------------------------------------------------------------------
### Create Plots ###
plot_solution(plot_dir, radius[], height[], L2[], V[], Pos[], equations)
#------------------------------------------------------------------------------
