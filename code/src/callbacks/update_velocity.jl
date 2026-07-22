# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    UpdateVelocityCallback

    Callback performs explicit time steps for the velocity v and pressure p0.
    If the callback is not used, both quantities will remain constant in time!

    CarpenterKennedy2N54() is used to perform the time integration for v and p0.
    The air velocity u(t,x) (i.e. "v1") is computed using the integrated v.

    Modifications compared to the TermiteMoundInducedAirflowTrixi.jl version:
        - Visual nodes are tracked in Callback cache
        - L2 error is computed and saved in Callback cache
        - Current solution state at visual nodes will be saved in an output.jl file
      
"""

mutable struct UpdateVelocityCallback{Vis_count, Vel_count, L2_count, L2_nodes, L2_error}
    a::Vis_count
    b::Vel_count
    c::L2_count
    d::L2_nodes
    e::L2_error
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:UpdateVelocityCallback})
    @nospecialize cb # reduce precompilation time

    update_velocity_callback = cb.affect!
    @unpack a, b = update_velocity_callback
    print(io, "UpdateVelocityCallback(a = ", a, ", b = ", b, ")")
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

function UpdateVelocityCallback( ;a=1::Int, b=1.0, c=1::Int, d=zeros(100,), e=0.0)
    # Convert plain real numbers to functions for unified treatment
    a_conv = isa(a, Real) ? Returns(a) : a
    b_conv = isa(b, Real) ? Returns(b) : b
    c_conv = isa(c, Real) ? Returns(c) : c
    d_conv = isa(d, Real) ? Returns(d) : d
    e_conv = isa(e, Real) ? Returns(e) : e
    update_velocity_callback = UpdateVelocityCallback{typeof(a_conv), typeof(b_conv), typeof(c_conv), typeof(d_conv), typeof(e_conv)}(a_conv, b_conv, c_conv, d_conv, e_conv)

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
    original_nodes = integrator.sol.prob.p.cache.elements.node_coordinates
    nodes = Float64.(vec(original_nodes[1, :, :]))
    L_nodes = length(nodes)
    t = LinRange(nodes[1], nodes[end], L_nodes)
    ti = t[1]:0.01:t[end]
    #-----------------------
    equations = integrator.sol.prob.p.equations
    tspan = integrator.sol.prob.tspan
    #-----------------------
    # TODO: use weights from integrator and remove FastGaussQuadrature dependence
    # try: write functions for quantities that require integration.
    # semi = integrator.sol.prob.p
    # integrate(f,semi)
    #
    # or: wrap_array to get elements for each variable
    # integrate_via_indices
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
    v1_dx_prev = [Interpolations.gradient(v1_prev_LI, nodes[i])[1] for i in 1:L_nodes]
    v_prev = v1_prev[1] * A(0, equations)
    Ti_prev = integrator.uprev[4:5:end-1]
    p0_prev = integrator.uprev[3:5:end-2]
    T_prev = p0_prev ./ rho_prev
    Tu_prev = [T_u(t_prev, x, y, equations) * equations.t_ref for (x, y) in zip(nodes, Ti_prev)] 
    #-----------------------
    # \frac{\partial p_0}{\partial t} = - \gamma p_0 \frac{\partial u}{\partial x} - \gamma \frac{\textrm{A}_x}{\textrm{A}} u p_0 - \frac{k_w}{\textrm{A} \sqrt{\textrm{A}}} \left(T - \textrm{T}_\textrm{u} \right)
    p0_dt_prev = .- equations.γ .* p0_prev .* v1_dx_prev .- equations.γ .* A_x_nodes ./ A_nodes .* v1_prev .* p0_prev .- I_w_nodes .* equations.k_w .* A_sqrt_A_inv .* (T_prev .- Tu_prev)
    p0_dt_prev = sum(p0_dt_prev) / length(p0_dt_prev)
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
        println(rho)
        error("NaN detected in rho!")
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
    p0_dt = .- equations.γ .* p0 .* v1_dx .- equations.γ .* A_x_nodes ./ A_nodes .* v1 .* p0 .- I_w_nodes .* equations.k_w .* A_sqrt_A_inv .* (T .- Tu)
    p0_dt = sum(p0_dt) / length(p0_dt)
    LI = LinearInterpolation([t_prev, t_now], [p0_dt_prev, p0_dt], extrapolation_bc=Line()) 
    prob = ODEProblem((u, p, t) -> LI(t), p0_prev[1], (t_prev, t_now))
    sol = solve(prob, CarpenterKennedy2N54(williamson_condition = false), dt = integrator.dt)
    p0 = sol.u[end]
    #-----------------------
    integrator.u[2:5:end-3] = v1
    integrator.u[3:5:end-2] .= p0
    #-----------------------
    a_now = update_velocity_callback.a()
    b_now = update_velocity_callback.b()
    c_now = update_velocity_callback.c()
    d_now = update_velocity_callback.d
    e_now = update_velocity_callback.e()
    a_now, b_now, c_now, d_now, e_now = if t_prev ≤ 0 && t_now > 0 
        (1, 1.0, 1, zeros(100,), 0.0)
    else
        (a_now, b_now, c_now, d_now, e_now)
    end
    a_new, b_new, c_new, d_new, e_new = write_plot_data(a_now, b_now, c_now, d_now, e_now, integrator, tspan, nodes, t, ti, 151, rho, v1, v1_prev, T, Ti, Tu, p0, equations)
    update_velocity_callback.a = isa(a_new, Real) ? Returns(a_new) : a_new
    update_velocity_callback.b = isa(b_new, Real) ? Returns(b_new) : b_new
    update_velocity_callback.c = isa(c_new, Real) ? Returns(c_new) : c_new
    update_velocity_callback.d = isa(d_new, Real) ? Returns(d_new) : d_new
    update_velocity_callback.e = isa(e_new, Real) ? Returns(e_new) : e_new
    #-----------------------
    return integrator
end

@inline function write_plot_data(a, b, c, d, e, integrator, tspan, nodes, t, ti, max_visnodes, rho, v1, v1_prev, T, Ti, Tu, p0, equations::TermiteMoundEquations1D)
    #-----------------------
    L2_index = range(tspan[1], tspan[end], 100)
    vis_index = range(tspan[1], tspan[end], max_visnodes)
    #-----------------------
    # L2 nest temperature error:
    b, c, d, e =    if integrator.tprev ≤ 0.0 && integrator.t > 0.0
                                        d[1] = T[1]
                                        b = sign(v1[1]) * abs(b)
                                        (b, 2, d, 0.0)
                                    else 
                                        (b,c,d,e)
                                    end
    #-----------------------
    b, c, d =   if integrator.t > L2_index[c]
                                d[c] = T[1]
                                b = if sign(b) == sign(v1[1])
                                    b
                                else 
                                    sign(v1[1]) * (abs(b) + 1)
                                end
                                (b,c + 1,d)
                else 
                    (b,c,d)
                end
    #-----------------------
    b, d, e =   if integrator.t == tspan[end]
                    d[end] = T[1]
                    b = if sign(b) == sign(v1[1])
                            abs(b) - 1.0
                        else 
                            (abs(b))
                        end
                    x_nodes = time2unscaled.(L2_index, equations) ./ 24.0
                    y_nodes = temp2unscaled.(d, equations)
                    L2_LI = LinearInterpolation(x_nodes, y_nodes)
                    p2_LI = LinearInterpolation([x_nodes[1], x_nodes[end]], [27.0, 27.0])
                    I = try 
                            sqrt( quadgk(x -> (L2_LI(x)-p2_LI(x))*(L2_LI(x)-p2_LI(x)), 0.0, x_nodes[end], rtol=1e-8)[1])
                        catch
                            100.0  
                        end
                    (b,d,I) 
                else 
                    (b,d,e)
                end
    #-----------------------
    b_new, c_new, d_new, e_new = (b,c,d,e)
    #-----------------------
    # save output:
    output_dir = joinpath(@__DIR__, "out//24h_termite_mound")
    vis_status =if integrator.tprev ≤ 0.0 && integrator.t > 0.0
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
        string1 = createString1(x_s, time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0)
        open(joinpath(output_dir, "output.jl"), "w") do file
                        write(file, string1)
                        flush(file)
        end
        1.0
    elseif integrator.t > vis_index[a]
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
        string2 = createString2(time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0) 
        open(joinpath(output_dir, "output.jl"), "a") do file
                        write(file, string2)
                        flush(file)
        end
        1.0
    elseif integrator.t == tspan[end]
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
        string2 = createString2(time2unscaled(integrator.t, equations), rho_s, v_s, T_s, Ti_s, Tu_s, p0) 
        string3 = createString3(e_new, b_new)
        open(joinpath(output_dir, "output.jl"), "a") do file
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
    return a_new, b_new, c_new, d_new, e_new
end

@inline function writeString(x,z)
    y = round.(x, digits=z) 
    y_str = join(y, ", ")  
    return "[" * y_str * "]" 
end

@inline function createString1(nodes, time, rho_s, v_s, T_s, Ti_s, Tu_s, p0)
    xnodes_str = writeString(nodes, 8)
    t_str = writeString(time, 8)
    rho_str = writeString(rho_s, 5)
    v_str = writeString(v_s, 5)
    T_str = writeString(T_s,5)
    Ti_str = writeString(Ti_s, 5)
    Tu_str = writeString(Tu_s, 5)
    p0_str = writeString(p0, 5)
    string =  "## output.jl ##\n#\n# 24h_Termite_Mound_sim\n#\n#------------------------------------------------------------------------------\n### read output ###\n@muladd @inline function read_output()\n    nodes = " * xnodes_str *"\n    #\n    t = " * t_str * "\n    rho = " * rho_str * "\n    v = " * v_str * "\n    T = " * T_str * "\n    Ti = " * Ti_str * "\n    Tu = " * Tu_str * "\n    p0 = " * p0_str * "\n    #\n"
    return string
end

@inline function createString2(time, rho_s, v_s, T_s, Ti_s, Tu_s, p0) 
    t_str = writeString(time, 8)
    rho_str = writeString(rho_s, 5)
    v_str = writeString(v_s, 5)
    T_str = writeString(T_s,5)
    Ti_str = writeString(Ti_s, 5)
    Tu_str = writeString(Tu_s, 5)
    p0_str = writeString(p0, 5)
    string = "    t =  vcat(t," * t_str *")\n    rho = vcat(rho," * rho_str * ")\n    v = vcat(v," * v_str * ")\n    T =  vcat(T," * T_str * ")\n    Ti =  vcat(Ti," * Ti_str * ")\n    Tu =  vcat(Tu," * Tu_str * ")\n    p0 =  vcat(p0," * p0_str * ")\n    #\n"
    return string
end   

@inline function createString3(x1, x2)
    string = "    L2_error = $(x1)\n    Vel_switches = $(x2)\n    return nodes, t, rho, v, T, Ti, Tu, p0, L2_error, Vel_switches\nend\n#------------------------------------------------------------------------------\n"
    return string
end  

end # @muladd
