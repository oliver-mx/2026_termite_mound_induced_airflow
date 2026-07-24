# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    UpdateVelocityCallback - winter

    Callback performs explicit time steps for the velocity v and pressure p0.
    If the callback is not used, both quantities will remain constant in time!

    CarpenterKennedy2N54() is used to perform the time integration for v and p0.
    The air velocity u(t,x) (i.e. "v1") is computed using the integrated v.

    Modifications compared to the TermiteMoundInducedAirflowTrixi.jl version:
        - GaussQuadrature weights precomputed
        - T_inside is computed at the final state
        - Final solution states saved in callback due to special design of f_main
        - Extrapolation between 24h simulations
"""

mutable struct UpdateVelocityCallback{Vis_count, Final_rho, Final_v1, Final_T, Final_Ti, Final_p0, Final_Tu, Final_T_inside}
    a::Vis_count
    rho_s::Final_rho
    v1_s::Final_v1
    T_s::Final_T
    Ti_s::Final_Ti
    p0_s::Final_p0
    Tu_s::Final_Tu
    T_inside_s::Final_T_inside
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:UpdateVelocityCallback})
    @nospecialize cb # reduce precompilation time

    update_velocity_callback = cb.affect!
    @unpack a, b = update_velocity_callback
    print(io, "UpdateVelocityCallback(a = ", a, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:UpdateVelocityCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        update_velocity_callback = cb.affect!

        setup = [
            "Vis Count" => update_velocity_callback.a#,
            #"Error" => update_velocity_callback.b
        ]
        Trixi.summary_box(io, "UpdateVelocityCallback", setup)
    end
end

function UpdateVelocityCallback( ;a=1::Int, rho_s=zeros(59,), v1_s=zeros(59,), T_s=zeros(59,), Ti_s=zeros(59,), p0_s=[1.0], Tu_s=zeros(59,), T_inside_s=[1.0])
    # Convert plain real numbers to functions for unified treatment
    a_conv = isa(a, Real) ? Returns(a) : a
    rho_conv = isa(rho_s, Real) ? Returns(rho_s) : rho_s
    v1_conv = isa(v1_s, Real) ? Returns(v1_s) : v1_s
    T_conv = isa(T_s, Real) ? Returns(T_s) : T_s
    Ti_conv = isa(Ti_s, Real) ? Returns(Ti_s) : Ti_s
    p0_conv = isa(p0_s, Real) ? Returns(p0_s) : p0_s
    Tu_conv = isa(Tu_s, Real) ? Returns(Tu_s) : Tu_s
    T_inside_conv = isa(T_inside_s, Real) ? Returns(T_inside_s) : T_inside_s
    update_velocity_callback = UpdateVelocityCallback{typeof(a_conv), typeof(rho_conv), typeof(v1_conv), typeof(T_conv), typeof(Ti_conv), typeof(p0_conv), typeof(Tu_conv), typeof(T_inside_conv)}(a_conv, rho_conv, v1_conv, T_conv, Ti_conv, p0_conv, Tu_conv, T_inside_conv)

    DiscreteCallback(condition, update_velocity_callback; save_positions = (false, false))
end

# callback always activated
@inline function condition(u, t, integrator)
    return true
end

# This method is called as callback during the time integration.
@inline function (update_velocity_callback::UpdateVelocityCallback)(integrator)
    """
    "update_velocity_callback" updates the velocity "v1" and pressure "p0" after each time step.
    This function is essential to solve the reformulated asymptotic model.

    Note: If the callback is not used, both "v1" and "p0" will remain constant in time!

    The CarpenterKennedy2N54() method is used to perform the time integration for v.
    Then, the air velocity u(t,x) (i.e. "v1") is computed using the integrated value of v.
    Finally, CarpenterKennedy2N54() is used again for "p0" and the state of the integrator is updated.
    """
    #-----------------------
    #println(fieldnames(typeof(integrator.sol.prob.p.solver.basis)))
    original_nodes = integrator.sol.prob.p.cache.elements.node_coordinates
    nodes = Float64.(vec(original_nodes[1, :, :]))
    L_nodes = length(nodes)
    t = LinRange(nodes[1], nodes[end], L_nodes)
    ti = t[1]:0.01:t[end]
    #-----------------------
    equations = integrator.sol.prob.p.equations
    tspan = integrator.sol.prob.tspan
    #-----------------------
    nodes_unique, idx_unique = unique_idx(nodes, equations)
    L_nodes_unique = length(nodes_unique)
    x, w = gausslegendre(L_nodes_unique)
    weights_unique = w./2
    weights = zeros(L_nodes)
    weights[idx_unique] = weights_unique
    #-----------------------
    A_nodes = [A(x, equations) for x in nodes]
    A_x_nodes = [A_x(x, equations) for x in nodes]
    A_nodes_inv = [inv(A(x, equations)) for x in nodes]
    A_sqrt_A_inv = [inv(sqrt(A(x, equations))) for x in nodes]
    I_w_nodes = [I_w(x, equations) for x in nodes]
    #-----------------------
    t_prev = integrator.tprev
    rho_prev = integrator.uprev[1:5:end-4]
    v1_prev = integrator.uprev[2:5:end-3]
    v1_prev_LI = bspline2linear(nodes, v1_prev, t, ti, equations)
    v1_dx_prev = [Interpolations.gradient(v1_prev_LI, nodes[i])[1] for i in 2:L_nodes-1]
    v1_dx_prev = vcat(Interpolations.gradient(v1_prev_LI, 0.0001)[1], vcat(v1_dx_prev, Interpolations.gradient(v1_prev_LI, 0.9999)[1]))
    v_prev = v1_prev[1] * A(0, equations)
    Ti_prev = integrator.uprev[4:5:end-1]
    p0_prev = integrator.uprev[3:5:end-2]
    T_prev = p0_prev ./ rho_prev
    Tu_prev = [T_u(t_prev, x, y, equations) * equations.t_ref for (x, y) in zip(nodes, Ti_prev)] 
    #-----------------------
    # \frac{\partial p_0}{\partial t} = - \gamma p_0 \frac{\partial u}{\partial x} - \gamma \frac{\textrm{A}_x}{\textrm{A}} u p_0 - \frac{k_w}{\textrm{A} \sqrt{\textrm{A}}} \left(T - \textrm{T}_\textrm{u} \right)
    #p0_dt_prev = - equations.γ * p0_prev[1] * v1_dx_prev[1] - equations.γ * A_x(0.0,equations) / A(0.0,equations) * v1_prev[1] * p0_prev[1] - I_w(0.0, equations) * equations.k_w / (A(0.0,equations) * sqrt(A(0.0,equations))) * (T_prev[1] - Tu_prev[1])
    p0_dt_prev = .- equations.γ .* p0_prev .* v1_dx_prev .- equations.γ .* A_x_nodes ./ A_nodes .* v1_prev .* p0_prev .- I_w_nodes .* equations.k_w .* A_sqrt_A_inv .* (T_prev .- Tu_prev)
    p0_dt_prev = sum((p0_dt_prev) .* weights)
    #-----------------------
    #\frac{\partial v}{\partial t} = \frac{1}{\int_0^1 \frac{\rho}{A} \, dy} \left[ \int_0^1 -\rho u u_x  - \rho u \left( \beta \eta - \beta\left(1-\eta\right) \vert u \vert \right) - \frac{\textrm{h}_x}{Fr^2}(\rho - \rho_{h_0}) \, dy \right]
    I_inv = inv.(sum((rho_prev.*A_nodes_inv) .* weights))
    beta = [equations.β ./ A(x,equations) for x in nodes]
    f = - rho_prev .* v1_prev .* v1_dx_prev - beta .* rho_prev .* v1_prev .* ( equations.η .- (1 - equations.η) .* abs.(v1_prev)) - h_x.(nodes, equations) .* (rho_prev .- equations.ρₕ₀) ./ equations.Fr²
    F = sum((f) .* weights)
    v_dt_prev = I_inv .* F
    #-----------------------
    t_now = integrator.t
    #-----------------------
    rho = integrator.u[1:5:end-4]
    if any(isnan, rho)
        #println(rho)
        error("NaN detected in rho!")
    end
    if t_prev == t_now
        error("Time step → 0!")
    end
    Ti = integrator.u[4:5:end-1]
    p0 = integrator.u[3:5:end-2]
    T = p0 ./ rho
    Tu = [T_u(t_now, x, y, equations) * equations.t_ref for (x, y) in zip(nodes, Ti)]
    #-----------------------
    ### Explixit velocity time step via CarpenterKennedy2N54() ###
    direction_prev = sign(v_prev)
    v_exp = v_prev + integrator.dt * v_dt_prev
    direction_exp = sign(v_exp)
    v_dt =  if direction_prev == direction_exp
            v1_exp = v1_prev .+ (v_exp - v_prev) / A(0.0, equations)
            v1_dx_exp = v1_dx_prev
            I_inv = inv.(sum((rho.*A_nodes_inv) .* weights))
            f = - rho .* v1_exp .* v1_dx_exp - beta .* rho .* v1_exp .* ( equations.η .- (1 - equations.η) .* abs.(v1_exp)) - h_x.(nodes, equations) .* (rho .- equations.ρₕ₀) ./ equations.Fr²
            F = sum((f) .* weights)
            I_inv .* F
            else
            v1_exp = reverse( - v1_prev .+ (v_exp + v_prev) / A(0.0, equations))
            v1_exp_LI = bspline2linear(nodes, v1_exp, t, ti, equations)
            v1_dx_exp = [Interpolations.gradient(v1_exp_LI, nodes[i])[1] for i in 1:L_nodes]
            I_inv = inv.(sum((rho.*A_nodes_inv) .* weights))
            f = - rho .* v1_exp .* v1_dx_exp - beta .* rho .* v1_exp .* ( equations.η .- (1 - equations.η) .* abs.(v1_exp)) - h_x.(nodes, equations).* (rho .- equations.ρₕ₀) ./ equations.Fr²
            F = sum((f) .* weights)
            I_inv .* F
    end
    LI = LinearInterpolation([t_prev, t_now], [v_dt_prev, v_dt], extrapolation_bc=Line()) 
    prob = ODEProblem((u, p, t) -> LI(t), v_prev, (t_prev, t_now))
    sol = solve(prob, CarpenterKennedy2N54(williamson_condition = false), dt = integrator.dt)
    v = sol.u[end]
    #-----------------------
    ### update u(t,x) ###
    Q = (- I_w_nodes .* equations.k_w .* A_sqrt_A_inv .* (T_prev .- Tu_prev)) ./ (equations.γ .* p0_prev[1])
    c = sum((Q.*A_nodes) .* weights)
    I_0 = cumsum((Q.*A_nodes.-c) .* weights)
    v1 = v .* A_nodes_inv .+ A_nodes_inv .* I_0
    v1_LI = LinearInterpolation2(nodes, v1, equations)
    v1_dx = [Interpolations.gradient(v1_LI, nodes[i])[1] for i in 1:L_nodes]
    #------------------------
    #p0_dt = - equations.γ * p0[1] * v1_dx[1] - equations.γ * A_x(0.0,equations) / A(0.0,equations) * v1[1] * p0[1] - I_w(0.0, equations) * equations.k_w / (A(0.0,equations) * sqrt(A(0.0,equations))) * (T[1] - Tu[1]) 
    p0_dt = .- equations.γ .* p0 .* v1_dx .- equations.γ .* A_x_nodes ./ A_nodes .* v1 .* p0 .- I_w_nodes .* equations.k_w .* A_sqrt_A_inv .* (T .- Tu)
    p0_dt = sum((p0_dt) .* weights)
    LI = LinearInterpolation([t_prev, t_now], [p0_dt_prev, p0_dt], extrapolation_bc=Line()) 
    prob = ODEProblem((u, p, t) -> LI(t), p0_prev[1], (t_prev, t_now))
    sol = solve(prob, CarpenterKennedy2N54(williamson_condition = false), dt = integrator.dt)
    p0 = sol.u[end]
    #-----------------------
    integrator.u[2:5:end-3] = v1
    integrator.u[3:5:end-2] .= p0
    #-----------------------
    if 0 == 1
        a_now = update_velocity_callback.a()
        a_now = if t_prev == tspan[1]
            1
        else
            a_now
        end
        a_new = write_plot_data(a_now, integrator, tspan, nodes, t, ti, 151, rho, v1, v1_prev, T, Ti, Tu, p0, equations)
        update_velocity_callback.a = isa(a_new, Real) ? Returns(a_new) : a_new
    end
    rho_final, v1_final, T_final, Ti_final, p0_final, Tu_final, T_inside_final =    if integrator.t == tspan[end]
                                                                                        callback_out(integrator, tspan, nodes, t, ti, rho, v1, T, Ti, Tu, p0, equations)
                                                                                    else
                                                                                        zeros(59,), zeros(59,), zeros(59,), zeros(59,), [0.0], zeros(59,), [0.0]
                                                                                    end
    update_velocity_callback.rho_s = isa(rho_final, Real) ? Returns(rho_final) : rho_final       
    update_velocity_callback.v1_s = isa(v1_final, Real) ? Returns(v1_final) : v1_final        
    update_velocity_callback.T_s = isa(T_final, Real) ? Returns(T_final) : T_final           
    update_velocity_callback.Ti_s = isa(Ti_final, Real) ? Returns(Ti_final) : Ti_final           
    update_velocity_callback.p0_s = isa(p0_final, Real) ? Returns(p0_final) : p0_final           
    update_velocity_callback.Tu_s = isa(Tu_final, Real) ? Returns(Tu_final) : Tu_final           
    update_velocity_callback.T_inside_s = isa(T_inside_final, Real) ? Returns(T_inside_final) : T_inside_final                                                                              
    #-----------------------
    return integrator
end

@inline function write_plot_data(a, integrator, tspan, nodes, t, ti, max_visnodes, rho, v1, v1_prev, T, Ti, Tu, p0, equations::PassiveHouseEquations1D)
    #-----------------------
    vis_index = range(tspan[1], tspan[end], max_visnodes)
    #-----------------------
    # save output:
    output_dir = joinpath(@__DIR__, "..//..//out//winter_simulation")
    vis_status = if integrator.tprev ≤ tspan[1] && integrator.t > tspan[1]
        day = floor((integrator.t + 0.01) * equations.tᵣ / 86400, digits=0)
        x_s = space2unscaled.(range(nodes[1], nodes[end], 59), equations)
        rho_LI = bspline2linear(nodes, rho, t, ti, equations)
        ρᵣ = 1.17
        rho_s = ρᵣ .* [rho_LI(x) for x in range(nodes[1], nodes[end], 59)]
        v1_LI = bspline2linear(nodes, v1, t, ti, equations)
        v_s = vel2unscaled.([v1_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        T_LI = bspline2linear(nodes, T, t, ti, equations)
        T_s = temp2unscaled.([T_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Ti_LI = bspline2linear(nodes, Ti, t, ti, equations)
        Ti_s = temp2unscaled.([Ti_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tu_LI = bspline2linear(nodes, Tu, t, ti, equations)
        Tu_s = temp2unscaled.([Tu_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tinside = [temp2unscaled( quadgk(x -> Ti_LI(x), equations.xa, equations.xc)[1] / (equations.xc - equations.xa), equations)]
        string1 = createString1(x_s, time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0, Tinside)
        filename = "output_$(day+1)"
        filename = filename[1:end-2] * ".jl"
        open(joinpath(output_dir, filename), "w") do file
                        write(file, string1)
                        flush(file)
        end
        1.0
    elseif integrator.t > vis_index[a]
        day = floor(integrator.t * equations.tᵣ / 86400, digits=0)
        rho_LI = bspline2linear(nodes, rho, t, ti, equations)
        ρᵣ = 1.17
        rho_s = ρᵣ .* [rho_LI(x) for x in range(nodes[1], nodes[end], 59)]
        v1_LI = bspline2linear(nodes, v1, t, ti, equations)
        v_s = vel2unscaled.([v1_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        T_LI = bspline2linear(nodes, T, t, ti, equations)
        T_s = temp2unscaled.([T_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Ti_LI = bspline2linear(nodes, Ti, t, ti, equations)
        Ti_s = temp2unscaled.([Ti_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tu_LI = bspline2linear(nodes, Tu, t, ti, equations)
        Tu_s = temp2unscaled.([Tu_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tinside = [temp2unscaled( quadgk(x -> Ti_LI(x), equations.xa, equations.xc)[1] / (equations.xc - equations.xa), equations)]
        string2 = createString2(time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0, Tinside) 
        filename = "output_$(day+1)"
        filename = filename[1:end-2] * ".jl"
        open(joinpath(output_dir, filename), "a") do file
                        write(file, string2)
                        flush(file)
        end
        1.0
    elseif integrator.t == tspan[end]
        day = floor((integrator.tprev) * equations.tᵣ / 86400, digits=0)
        rho_LI = bspline2linear(nodes, rho, t, ti, equations)
        ρᵣ = 1.17
        rho_s = ρᵣ .* [rho_LI(x) for x in range(nodes[1], nodes[end], 59)]
        v1_LI = bspline2linear(nodes, v1, t, ti, equations)
        v_s = vel2unscaled.([v1_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        T_LI = bspline2linear(nodes, T, t, ti, equations)
        T_s = temp2unscaled.([T_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Ti_LI = bspline2linear(nodes, Ti, t, ti, equations)
        Ti_s = temp2unscaled.([Ti_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tu_LI = bspline2linear(nodes, Tu, t, ti, equations)
        Tu_s = temp2unscaled.([Tu_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
        Tinside = [temp2unscaled( quadgk(x -> Ti_LI(x), equations.xa, equations.xc)[1] / (equations.xc - equations.xa), equations)]
        string2 = createString2(time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0, Tinside) 
        string3 = createString3()
        filename = "output_$(day+1)"
        filename = filename[1:end-2] * ".jl"
        open(joinpath(output_dir, filename), "a") do file
                        write(file, string2 * string3)
                        flush(file)
        end
        1.0
    else
       0.0
    end
    #-----------------------
    if vis_status == 1.0
        a_new = a + 1
    else
        a_new = a
    end
    #-----------------------
    return a_new
end

@inline function callback_out(integrator, tspan, nodes, t, ti, rho, v1, T, Ti, Tu, p0, equations::PassiveHouseEquations1D)
    ρᵣ = 1.17
    rho_LI = bspline2linear(nodes, rho, t, ti, equations)
    v1_LI = bspline2linear(nodes, v1, t, ti, equations) 
    T_LI = bspline2linear(nodes, T, t, ti, equations)
    Ti_LI = bspline2linear(nodes, Ti, t, ti, equations)
    Tu_LI = bspline2linear(nodes, Tu, t, ti, equations)
    rho_s = ρᵣ .* [rho_LI(x) for x in range(nodes[1], nodes[end], 59)]
    v_s = vel2unscaled.([v1_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
    T_s = temp2unscaled.([T_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
    Ti_s = temp2unscaled.([Ti_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
    p0_s = [integrator.u[3]]
    Tu_s = temp2unscaled.([Tu_LI(x) for x in range(nodes[1], nodes[end], 59)], equations)
    Tinside = [temp2unscaled( quadgk(x -> Ti_LI(x), equations.xa, equations.xc)[1] / (equations.xc - equations.xa), equations)]
    #-----------------------
    return rho_s, v_s, T_s, Ti_s, p0_s, Tu_s, Tinside
end

@inline function writeString(x,z)
    y = round.(x, digits=z) 
    y_str = join(y, ", ")  
    return "[" * y_str * "]" 
end

@inline function createString1(nodes, time, rho_s, v_s, T_s, Ti_s, p0, Tu_s, Tinside)
    xnodes_str = writeString(nodes, 8)
    t_str = writeString(time, 8)
    rho_str = writeString(rho_s, 5)
    v_str = writeString(v_s, 5)
    T_str = writeString(T_s,5)
    Ti_str = writeString(Ti_s, 5)
    Tu_str = writeString(Tu_s, 5)
    p0_str = writeString(p0, 5)
    Tinside_str = writeString(Tinside, 5)
    string =  "## output.jl ##\n#\n# Winter_simulation\n#\n#------------------------------------------------------------------------------\n### read output ###\n@muladd @inline function read_output()\n    nodes = " * xnodes_str *"\n    #\n    t = " * t_str * "\n    rho = " * rho_str * "\n    v = " * v_str * "\n    T = " * T_str * "\n    Ti = " * Ti_str * "\n    p0 = " * p0_str * "\n    Tu = " * Tu_str * "\n    Tinside = " * Tinside_str * "\n    #\n"
    return string
end

@inline function createString2(time, rho_s, v_s, T_s, Ti_s, p0, Tu_s, Tinside) 
    t_str = writeString(time, 8)
    rho_str = writeString(rho_s, 5)
    v_str = writeString(v_s, 5)
    T_str = writeString(T_s,5)
    Ti_str = writeString(Ti_s, 5)
    Tu_str = writeString(Tu_s, 5)
    p0_str = writeString(p0, 5)
    Tinside_str = writeString(Tinside, 5)
    string = "    t =  vcat(t," * t_str *")\n    rho = vcat(rho," * rho_str * ")\n    v = vcat(v," * v_str * ")\n    T =  vcat(T," * T_str * ")\n    Ti =  vcat(Ti," * Ti_str * ")\n    p0 =  vcat(p0," * Tu_str * ")\n    Tu =  vcat(Tu," * p0_str * ")\n    Tinside =  vcat(Tinside," * Tinside_str * ")\n    #\n"
    return string
end   

@inline function createString3()
    string = "    return nodes, t, rho, v, T, Ti, p0, Tu Tinside\nend\n#------------------------------------------------------------------------------\n"
    return string
end  

@inline function NextVals(t, z1, z2, z3, z4, z5, z6, y7, z7)
    q1 = z1
    q2 = z2
    q3 = z3
    q5 = z5
    q6 = z6
    q7 = newVals(t, y7, z7)
    q4 = z4 .+ (q7 - z7)
    return q1, q2, q3, q4, q5, q6, q7
end

@inline function newVals(t, ya, yb)
    L = length(ya)
    yc = if L == 1
            linexp(t, ya, yb)
        else
            [linexp(t, ya[i], yb[i]) for i in 1:L]
        end
    return yc
end

@inline function linexp(t, ya, yb)
    return ya + (yb - ya) / (t[2] - t[1]) * (t[3] - t[1])
end

end # @muladd
