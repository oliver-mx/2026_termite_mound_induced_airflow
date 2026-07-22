# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
PassiveHouseEquations1D(;γ=1.3987529976019184, k_i=2.2216666666666667e-5, k_w=0.12601678420854714, k_s=0.0002551669105616394, q₀=300.0, tᵣ=107.5, uᵣ=0.2, xa=0.13953488372093023, xb=0.37209302325581395, xc=0.8604651162790697, h=5.5, L=21.5, β=0.3309860294294887, η=0.93, Fr²=0.00018964985894791742, ρₕ₀=0.982, T_ref=297.76980029566033, t_ref=0.0033582989242263127, T0=27.0, v0=-2.219, Ti_LI=LinearInterpolation([0.0, 0.372, 0.373, 1.0],[24.8, 25.0, 18.0, 17.5]))

PassiveHouseEquations1D (PHE) in one space dimension. The equations are given by
```math
    \begin{equation}
    \frac{\partial \rho}{\partial t} + \frac{\partial \left( \rho u\right)}{\partial x} = - \frac{\textrm{A}_x}{\textrm{A}}\rho u 
    \end{equation}
    \begin{equation}
    \frac{\partial v}{\partial t} = \frac{1}{\int_0^1 \frac{\rho}{A} \, dy} \left[ 
    \int_0^1 -\rho u u_x  - \rho u \left( \beta \eta - \beta\left(1-\eta\right) \vert u \vert \right) - \frac{\textrm{h}_x}{Fr^2}(\rho - \rho_{h_0}) \, dy \right] 
    \end{equation}
    \begin{equation}
    \frac{\partial p_0}{\partial t} + \gamma p_0 \frac{\partial u}{\partial x}= - \gamma \frac{\textrm{A}_x}{\textrm{A}} u p_0 - \frac{k_w}{\textrm{A} \sqrt{\textrm{A}}} \left(T - \textrm{T}_\textrm{u} \right) 
    \end{equation}
    \begin{equation}
    \frac{\partial T_i}{\partial t} = k_i \left( T - T_i \right)  + k_s \textrm{q}_{\textrm{s}} 
    \end{equation}
    \begin{equation}
    p_0 = \rho T 
    \end{equation}
    \begin{equation}
    u(t,x) = \frac{v(t)}{A(x)} + \frac{1}{A(x)} \int_0^x A(y)\left[ -\frac{\partial  p_0}{\partial t} - \frac{k_w}{\textrm{A} \sqrt{\textrm{A}}} \left(T - \textrm{T}_\textrm{u} \right)\right] \, dy 
    \end{equation}
```
The unknown quantities of the TME are the air density ``ρ``, the air velocity ``u``, the leading order pressure ``p_0``, the internal temperature of the mound ``T_i``, and the auxiliary velocity variable ``v``.
Due to the single spatial dimensional, the velocity ``u`` is referred to as ``v1`` in the code.
We denote the (possibly) variable cross section function ``A(x)``, the height profile function ``h(x)`` and the boundary temperature function ``T_u(t,x)``.
The boundary temperature function is defined by
```math
    \begin{align}
    \textrm{T}_\textrm{u}(t,x)  = \begin{cases}
    \textrm{T}_\textrm{soil}(t) & ,\text{if } x \in [0,x_1] \, \cup \, (x_3, 1] \\
    \frac{\textrm{T}_\textrm{air}(t) + T_i(t,x) }{2} & ,\text{if } x \in (x_1,x_3] \\
    \end{cases}
    \end{align}
```
It depends on the internal mound temperature ``T_i``, the soil temperature (at 30cm depth) ``T_{soil}`` and the ambient air temperature ``T_{air}``. 
The parameters ``x_1``, ``x_2`` and ``x_3`` are fixed spatial locations along the flow channel.

The viscosity parameter is denoted by ``η`` and the dimensionless parameters are ``k_w`` and ``k_i``.

References for the TME and PHE:
- Marx, Oliver P. and Gasser, Ingenuin and Schmidgall Annika (2026)
...
[DOI: ...](https://doi.org/...)

Modifications compared to the TermiteMoundInducedAirflowTrixi.jl version:
- Conversion function for scaled and unscaled quantities.
- Varying ``T_{soil}``, ``T_{air}`` and ``q`` across 150 days
"""

struct PassiveHouseEquations1D{RealT <: Real} <: AbstractEquations{1, 5}
γ::RealT
inv_gamma_minus_one::RealT
k_i::RealT 
k_w::RealT 
k_s::RealT
q₀::RealT
tᵣ::RealT
uᵣ::RealT
xa::RealT
xb::RealT
xc::RealT
h::RealT
L::RealT
β::RealT 
η::RealT 
Fr²::RealT 
ρₕ₀::RealT 
T_ref::RealT
t_ref::RealT
ρ_LI::Interpolations.Extrapolation 
v_LI::Interpolations.Extrapolation 
T_LI::Interpolations.Extrapolation 
Ti_LI::Interpolations.Extrapolation 
p0::RealT
    function PassiveHouseEquations1D(; γ=1.3987529976019184, k_i=2.2216666666666667e-5, k_w=0.12601678420854714, k_s=0.0003402225474155192, q₀=400.0, tᵣ=107.5, uᵣ=0.2, xa=0.13953488372093023, xb=0.37209302325581395, xc=0.8604651162790697, h=5.5, L=21.5, β=0.3309860294294887, η=0.93, Fr²=0.00018964985894791742, ρₕ₀=0.982, T_ref=297.76980029566033, t_ref=0.0033582989242263127, ρ_LI=LinearInterpolation([0.0, 1.0],1.17.*inv.(temp2scaled.([27.0, 27.0]))), v_LI=LinearInterpolation([0.0, 1.0],[-2.219, -2.219]), T_LI=LinearInterpolation([0.0, 1.0],[27.0, 27.0]), Ti_LI=LinearInterpolation([0.0, 0.372, 0.373, 1.0],[24.8, 25.0, 18.0, 17.5]), p0 = 1.0)
        γ, inv_gamma_minus_one = promote(γ, inv(γ - 1.0))
        new{typeof(β)}(γ, inv_gamma_minus_one, k_i, k_w, k_s, q₀, tᵣ, uᵣ, xa, xb, xc, h, L, β, η, Fr², ρₕ₀, T_ref, t_ref, ρ_LI, v_LI, T_LI, Ti_LI, p0)
    end
end

function Trixi.varnames(::typeof(cons2cons), equations::PassiveHouseEquations1D)
    ("rho", "v1", "p0", "Ti", "x_var")
end

function Trixi.varnames(::typeof(cons2prim), equations::PassiveHouseEquations1D)
    ("rho", "v1", "p0", "Ti", "x_var")
end

function Trixi.varnames(::typeof(prim2cons), equations::PassiveHouseEquations1D)
    ("rho", "v1", "p0", "Ti", "x_var")
end

function Trixi.varnames(::typeof(cons2entropy), equations::PassiveHouseEquations1D)
    ("rho", "v1", "p0", "Ti", "x_var")
end

function Trixi.varnames(::typeof(cons2cons), equations::PassiveHouseEquations1D)
    ("rho", "v1", "p0", "Ti", "x_var")
end

function Trixi.cons2prim(u, ::PassiveHouseEquations1D)
    return u 
end

function Trixi.prim2cons(prim, ::PassiveHouseEquations1D)
    return prim 
end

function Trixi.cons2entropy(u, ::PassiveHouseEquations1D)
    return u 
end

function Trixi.cons2cons(u, ::PassiveHouseEquations1D)
    return u
end

@inline function PassiveHouseInitialCondition(x, t, equations::PassiveHouseEquations1D)
    RealT = eltype(x)
    ρ_LI = equations.ρ_LI
    v_LI = equations.v_LI
    Ti_LI = equations.Ti_LI
    rho = ρ_LI(x[1]) ./ 1.17
    v1 = vel2scaled(v_LI(x[1]), equations)
    Ti = temp2scaled(Ti_LI(x[1]), equations)
    p0 = equations.p0[]
    return SVector(rho, v1, p0, Ti, x[1])
end

@inline function source_terms(u, x, t, equations::PassiveHouseEquations1D)
    rho, v1, p0, Ti, _ = u
    x_var = x[1]
    T = p0 / rho
    du1 = - A_x(x_var, equations) / A(x_var, equations) * rho * v1
    du4 = equations.k_i * (T - Ti) + equations.k_s * Q_s(t, x_var, equations)
    return SVector(du1, 0.0, 0.0, du4, 0.0)
end

@inline function temp2scaled(y,equations::PassiveHouseEquations1D)
    z = 273.15
    y = y .+ z
    x = y ./ equations.T_ref
    return x
end

@inline function vel2scaled(y, equations::PassiveHouseEquations1D)
    x = inv(equations.uᵣ*100) .* y 
    return x
end

@inline function time2scaled(y, equations::PassiveHouseEquations1D)
    return y * (3600/equations.tᵣ)
end

@inline function space2unscaled(x, equations::PassiveHouseEquations1D)
    return equations.L .* x 
end

@inline function temp2unscaled(x, equations::PassiveHouseEquations1D)
    z = 273.15
    return x .* equations.T_ref .- z
end

@inline function vel2unscaled(x, equations::PassiveHouseEquations1D)
    return x.* equations.uᵣ * 100
end

@inline function time2unscaled(x, equations::PassiveHouseEquations1D)
    return x ./ (3600/equations.tᵣ)
end

@inline function h(x, equations::PassiveHouseEquations1D)
    k = 35.0
    x1 = 0.2558139534883721
    x2 = 0.5116279069767442
    x3 = 0.7558139534883721
    A1 = 0.0
    A2 = 5.0 / equations.L
    A3 = 4.5 / equations.L
    s1 = 1 / (1 + exp(-k * (x - x1)))
    s2 = 1 / (1 + exp(-k * (x - x2)))
    s3 = 1 / (1 + exp(-k * (x - x3)))
    y = A1 * (1 - s1) + A2 * (s1 - s2) + A3 * (s2 - s3)
    return y
end

@inline function h_x(x, equations::PassiveHouseEquations1D)
    k = 35.0
    x1 = 0.2558139534883721
    x2 = 0.5116279069767442
    x3 = 0.7558139534883721
    A1 = 0.0
    A2 = 5.0 / equations.L
    A3 = 4.5 / equations.L
    ds1 = k * exp(-k * (x - x1)) / ((1 + exp(-k * (x - x1)))*(1 + exp(-k * (x - x1))))
    ds2 = k * exp(-k * (x - x2)) / ((1 + exp(-k * (x - x2)))*(1 + exp(-k * (x - x2))))
    ds3 = k * exp(-k * (x - x3)) / ((1 + exp(-k * (x - x3)))*(1 + exp(-k * (x - x3))))
    dy = -A1 * ds1 + A2 * (ds1 - ds2) + A3 * (ds2 - ds3)
    return dy
end

@inline function A(x, equations::PassiveHouseEquations1D)
    k = 50.0
    x1 = 0.13953488372093023
    x2 = 0.37209302325581395
    x3 = 0.6511627906976745
    x4 = 0.8604651162790697
    A1 = 1.0
    A2 = 2.179 / 3.9
    A3 = 1.078 / 3.9
    A4 = A2
    s1 = 1 / (1 + exp(-k * (x - x1)))
    s2 = 1 / (1 + exp(-k * (x - x2)))
    s3 = 1 / (1 + exp(-k * (x - x3)))
    s4 = 1 / (1 + exp(-k * (x - x4)))
    s5 = 1 / (1 + exp(-k * (x - 1.0)))
    y = A1 * (1 - s1) + A2 * (s1 - s2) + A3 * (s2 - s3) + A4 * (s3 - s4) + A1 * (s4 - s5) + A1 * s5
    return y
end

@inline function A_x(x, equations::PassiveHouseEquations1D)
    k = 50.0
    x1 = 0.13953488372093023
    x2 = 0.37209302325581395
    x3 = 0.6511627906976745
    x4 = 0.8604651162790697
    A1 = 1.0
    A2 = 2.179 / 3.9
    A3 = 1.078 / 3.9
    A4 = A2
    ds1 = k * exp(-k * (x - x1)) / ((1 + exp(-k * (x - x1)))*(1 + exp(-k * (x - x1))))
    ds2 = k * exp(-k * (x - x2)) / ((1 + exp(-k * (x - x2)))*(1 + exp(-k * (x - x2))))
    ds3 = k * exp(-k * (x - x3)) / ((1 + exp(-k * (x - x3)))*(1 + exp(-k * (x - x3))))
    ds4 = k * exp(-k * (x - x4)) / ((1 + exp(-k * (x - x4)))*(1 + exp(-k * (x - x4))))
    ds5 = k * exp(-k * (x - 1.0)) / ((1 + exp(-k * (x - 1.0)))*(1 + exp(-k * (x - 1.0))))
    dy = -A1 * ds1 + A2 * (ds1 - ds2) + A3 * (ds2 - ds3) + A4 * (ds3 - ds4) + A1 * (ds4 - ds5) + A1 * ds5
    return dy
end

@inline function T_u(t_var, x_var, Ti, equations::PassiveHouseEquations1D)
    ϵ0 = 0.5
    ϵ1 = 0.1
    t_var = mod(t_var, 86400/equations.tᵣ) / (86400/equations.tᵣ/24)
    c0 = T_air(t_var, equations)
    c1 = T_soil(t_var, equations)
    c2 = ϵ0 * c0 + (1.0-ϵ0) * Ti * equations.T_ref 
    c3 = ϵ1 * c0 + (1.0-ϵ1) * Ti * equations.T_ref 
    return ifelse( (x_var ≤ equations.xa) || (x_var ≥ equations.xc), c1, ifelse(x_var ≤ equations.xb, c2, c3))
end

@inline function T_u_dt(t_var, x_var, Ti_dt, equations::PassiveHouseEquations1D)
    ϵ0 = 0.5 
    ϵ1 = 0.1
    t_var = mod(t_var, 86400/equations.tᵣ) / (86400/equations.tᵣ/24)
    c0_dt = dt_T_air(t_var, equations)
    c1_dt = dt_T_soil(t_var, equations)
    c2_dt = ϵ0 * c0_dt + (1.0-ϵ2) * Ti_dt
    c3_dt = ϵ1 * c0_dt + (1.0-ϵ3) * Ti_dt
    return ifelse( (x_var ≤ equations.xa) || (x_var ≥ equations.xc), c1_dt, ifelse(x_var ≤ equations.xb, c2_dt, c3_dt))
end

@inline function LinearInterpolation2(x, y, equations::PassiveHouseEquations1D)
    iunique_indices = unique!(collect(zip(x, y)))[2]
    x_unique = []
    y_unique = []
    seen = Dict{typeof(x[1]), Bool}()
    for (xi, yi) in zip(x, y)
        if haskey(seen, xi)
            continue
        else
            push!(x_unique, xi)
            push!(y_unique, yi)
            seen[xi] = true
        end
    end
    LI = LinearInterpolation(x_unique, y_unique)
    return LI
end

@inline function Fourir(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, t_var, equations::PassiveHouseEquations1D)
    return a0 + a1 * cos(pi * t_var / 12.0) + a2 * sin(pi * t_var / 12.0) + a3 * cos(pi * t_var / 6.0) + a4 * sin(pi * t_var / 6.0) + a5 * cos(pi * t_var / 4.0) + a6 * sin(pi * t_var / 4.0) + a7 * cos(pi * t_var / 3.0) + a8 * sin(pi * t_var / 3.0) + a9 * cos(pi * t_var / 2.4) + a10 * sin(pi * t_var /2.4) 
end

@inline function dt_Fourir(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, t_var, equations::PassiveHouseEquations1D)
    return - a1 .* sin(pi * t_var / 12.0) .* (pi / 12.0) + a2 * cos(pi * t_var / 12.0) * (pi / 12.0) - a3 * sin(pi * t_var / 6.0) * (pi / 6.0) + a4 * cos(pi * t_var / 6.0) * (pi / 6.0) - a5 * sin(pi * t_var / 4.0) * (pi / 4.0) + a6 * cos(pi * t_var / 4.0) * (pi / 4.0) - a7 * sin(pi * t_var / 3.0) * (pi / 3.0) + a8 * cos(pi * t_var / 3.0) * (pi / 3.0) - a9 * sin(pi * t_var / 2.4) * (pi / 2.4) + a10 * cos(pi * t_var / 2.4) .* (pi / 2.4)              
end

@inline function bspline2linear(nodes, vals, t, ti, equations::PassiveHouseEquations1D)
    itp = Interpolations.scale(interpolate(hcat(nodes, vals), (BSpline(Cubic(Natural(OnGrid()))), NoInterp())), t, 1:2)
    nodesitp, valsitp = [itp(t,1) for t in ti], [itp(t,2) for t in ti]
    return LinearInterpolation(nodesitp, valsitp)
end

@muladd @inline function T_air(t_var, equations::PassiveHouseEquations1D)
    winter = 150 * 803.7209302325581
    # Mean 4.8 °C
    # abs(4.8 - 12.7)/2 = 3.9499999999999997
    #
    t_var_2 = mod(t_var, 86400/equations.tᵣ) / (86400/equations.tᵣ/24)
    #
    T_air_winter = -3.9499999999999997*sin((t_var_2+4)*2*pi/24) + 277.95
    # Mean 10.5 °C
    # abs(10.5 - 17.3)/2 = 3.4000000000000004
    y = 5.7 - 5.7*sin(t_var * pi / winter) + T_air_winter
    # https://www.wetterkontor.de/wetter-rueckblick/monats-und-jahreswerte.asp?id=80&jr0=2024&jr1=2025&mo0=1&mo1=12 
   return y
end

@muladd @inline function dt_T_air(t_var, equations::PassiveHouseEquations1D)
    winter = 150 * 803.7209302325581
    t_var_2 = mod(t_var, 86400/equations.tᵣ) / (86400/equations.tᵣ/24)
    dt_air_winter = -3.9499999999999997 * (2*pi/24) * cos((t_var_2+4)*2*pi/24)
    dy = -5.7*pi/winter*cos(t_var * pi / winter) + dt_air_winter
    return dy
end

@inline function T_soil(t_var, equations::PassiveHouseEquations1D)
    return 284.15
end

@inline function dt_T_soil(t_var, equations::PassiveHouseEquations1D)
    return 0.0
end

@inline function I_w(x_var, equations::PassiveHouseEquations1D)
    1.0
end

@inline function I_s(x_var, equations::PassiveHouseEquations1D)
    LI = LinearInterpolation([0.0, equations.xa - 1e-8, equations.xa, equations.xb, equations.xb + 1e-8, 1.0],[0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    return LI(x_var)
end

@inline function Q_s(t_var, x_var, equations::PassiveHouseEquations1D)
    winter = 150 * 803.7209302325581
    t_var_2 = mod(t_var, 86400/equations.tᵣ) / (86400/equations.tᵣ) 
    t_now = mod(t_var, 803.7209302325581) / 803.7209302325581
    y = 1.0 - 0.75*(sin(t_var * pi / winter))
    return I_s(x_var, equations) * (y * q_s(t_now, equations) .- 10/equations.q₀)
end

@inline function q_s(t_var, equations)
    z = equations.q₀
    xs = [0, 7, 10.5, 12, 13, 15, 16.5, 18.5, 24] ./ 24;
    ys = (10 .+ [-10, -10, 0.3*z, 0.6 * z, z, z, 0, -10, -10]) ./ z;
    LI = LinearInterpolation(xs, ys);
    return LI(t_var)
end

@inline function unique_idx(x, equations::PassiveHouseEquations1D)
    y=1:1:length(x)
    iunique_indices = unique!(collect(zip(x, y)))[2]
    x_unique = Vector{typeof(x[1])}()
    y_unique = Vector{Int}()
    seen = Dict{typeof(x[1]), Bool}()
    for (xi, yi) in zip(x, y)
        if haskey(seen, xi)
            continue
        else
            push!(x_unique, xi)
            push!(y_unique, yi)
            seen[xi] = true
        end
    end
    return x_unique, y_unique
end

# Calculate 1D flux for a single point
@inline function Trixi.flux(u, orientation::Integer, equations::PassiveHouseEquations1D)
    rho, v1, _, _, _ = u
    f1 = rho * v1
    return SVector(f1, 0.0, 0.0, 0.0, 0.0)
end

@inline function flux_ranocha(u_ll, u_rr, orientation::Integer, equations::PassiveHouseEquations1D)
    rho_ll, v1_ll, _, _, _ = u_ll
    rho_rr, v1_rr, _, _, _ = u_rr
    rho_mean = ln_mean(rho_ll, rho_rr)
    v1_mean =   if v1_ll > 0.0 && v1_rr > 0.0
                    ln_mean(v1_ll, v1_rr)
                elseif v1_ll < 0.0 && v1_rr < 0.0
                    - ln_mean(-v1_ll, -v1_rr)
                else
                    0.5f0 * (v1_ll + v1_rr)
                end
    f1 = rho_mean * v1_mean
    return SVector(f1, 0.0, 0.0, 0.0, 0.0)
end

@inline function Trixi.max_abs_speeds(u, equations::PassiveHouseEquations1D)
    rho, v1, p0, _, _ = u
    c = sqrt(equations.γ * p0 / rho) 
    return (abs(v1) + c,)
end

end # @muladd
