module RayTraceGR

using DifferentialEquations
using Images
using LinearAlgebra
using StaticArrays



export Dual
struct Dual{T,DT} <: Real
    val::T
    eps::DT
end

function Dual{T,DT}(val::T) where {T,DT}
    Dual{T,DT}(val, zeros(DT))
end
function Dual{T,DT}(val) where {T,DT}
    Dual{T,DT}(val, zeros(DT))
end

Base.promote_rule(::Type{Dual{T,DT}}, ::Type{T}) where {T,DT} = Dual{T,DT}
Base.promote_rule(::Type{Dual{T,DT}}, ::Type{<:Integer}) where {T,DT} =
    Dual{T,DT}
Base.promote_rule(::Type{Dual{T,DT}}, ::Type{Dual{U,DU}}) where {T,DT,U,DU} =
    Dual{promote{T,U}, promote{DT,DU}}

# Cannot have a return type annotation here, as this would lead to an
# infinite recursion
function Base.convert(::Type{Dual{T,DT}}, x::Dual{U,DU}) where {T,DT,U,DU}
    Dual{T,DT}(convert(T, x.val), convert(DT, x.eps))
end
function Base.convert(::Type{Dual{T,DT}}, val::Number)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(convert(T, val))
end

function Base.eps(::Type{Dual{T,DT}})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(eps(T))
end

function Base.zero(::Type{Dual{T,DT}})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(zero(T))
end
function Base.one(::Type{Dual{T,DT}})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(one(T))
end

function Base.:+(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(+x.val, +x.eps)
end

function Base.:-(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(-x.val, -x.eps)
end

function Base.:+(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val+y.val, x.eps+y.eps)
end
function Base.:+(x::Dual{T,DT}, a::T)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val+ a, x.eps)
end
function Base.:+(a::T, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a+x.val, x.eps)
end
function Base.:+(x::Dual{T,DT}, a::Integer)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val+a, x.eps)
end
function Base.:+(a::Integer, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a+x.val, x.eps)
end

function Base.:-(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val-y.val, x.eps-y.eps)
end
function Base.:-(x::Dual{T,DT}, a::T)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val- a, x.eps)
end
function Base.:-(a::T, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a-x.val, x.eps)
end
function Base.:-(x::Dual{T,DT}, a::Integer)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val-a, x.eps)
end
function Base.:-(a::Integer, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a-x.val, x.eps)
end

function Base.:*(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val*y.val, x.eps.*y.val + x.val.*y.eps)
end
function Base.:*(x::Dual{T,DT}, a::T)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val*a, x.eps.*a)
end
function Base.:*(a::T, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a*x.val, a.*x.eps)
end
function Base.:*(x::Dual{T,DT}, a::Integer)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val*a, x.eps.*a)
end
function Base.:*(a::Integer, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a*x.val, a.*x.eps)
end

# 1/(a+ϵb) = (a-ϵb)/a^2
function Base.inv(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(inv(x.val), -inv(x.val)^2 .* x.eps)
end

function Base.:/(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val/y.val, (x.eps * y.val - x.val * y.eps) / y.val^2)
end
function Base.:/(x::Dual{T,DT}, a::T)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val/a, x.eps./a)
end
function Base.:/(x::Dual{T,DT}, a::Integer)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val/a, x.eps./a)
end

function Base.:\(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val\y.val, x.val^2 \ (x.eps .* y.val - x.val .* y.eps))
end
function Base.:\(a::T, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a\x.val, a.\x.eps)
end
function Base.:\(a::Integer, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a\x.val, a.\x.eps)
end

function Base.:^(x::Dual{T,DT}, n::Integer)::Dual{T,DT} where {T,DT}
    n == 0 && return one(x)
    Dual{T,DT}(x.val^n, n * x.val^(n-1) .* x.eps)
end
function Base.:^(x::Dual{T,DT}, a::T)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val^a, a * x.val^(a-1) .* x.eps)
end
function Base.:^(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    r = x.val ^ y.val
    Dual{T,DT}(r, y.val/x.val * r .* x.eps + r * log(x.val) .* y.eps)
end

function Base.abs(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(abs(x.val), copysign(one(T), x.val) .* x.eps)
end

function Base.acos(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(acos(x.val), -1 / sqrt(1 - x.val^2) .* x.eps)
end

function Base.asin(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(asin(x.val), 1 / sqrt(1 - x.val^2) .* x.eps)
end

function Base.atan(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(atan(x.val), 1 / (1 + x.val^2) .* x.eps)
end
function Base.atan(y::Dual{T,DT}, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    ρ2 = x.val^2 + y.val^2
    Dual{T,DT}(atan(y.val, x.val), x.val .* y.eps - y.val / ρ2 .* x.eps)
end

function Base.cbrt(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    r = cbrt(x.val)
    Dual{T,DT}(r, r/(3*x.val) .* x.eps)
end

function Base.cos(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(cos(x.val), -sin(x.val) .* x.eps)
end

function Base.exp(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    r = exp(x.val)
    Dual{T,DT}(r, r .* x.eps)
end

function Base.log(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(log(x.val), 1 / x.val .* x.eps)
end

function Base.sin(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(sin(x.val), cos(x.val) .* x.eps)
end

function Base.sqrt(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    r = sqrt(x.val)
    Dual{T,DT}(r, 1/(2*r) .* x.eps)
end

function Base.:(==)(x::Dual{T,DT}, y::Dual{T,DT})::Bool where {T,DT}
    x.val == y.val
end
function Base.:(==)(x::Dual{T,DT}, a::T)::Bool where {T,DT}
    x.val == a
end
function Base.:(==)(x::Dual{T,DT}, a::Integer)::Bool where {T,DT}
    x.val == a
end
function Base.:(==)(a::T, x::Dual{T,DT})::Bool where {T,DT}
    a == x.val
end
function Base.:(==)(a::Integer, x::Dual{T,DT})::Bool where {T,DT}
    a == x.val
end

function Base.:(<)(x::Dual{T,DT}, y::Dual{T,DT})::Bool where {T,DT}
    x.val < y.val
end
function Base.:(<)(x::Dual{T,DT}, a::T)::Bool where {T,DT}
    x.val < a
end
function Base.:(<)(x::Dual{T,DT}, a::Integer)::Bool where {T,DT}
    x.val < a
end
function Base.:(<)(a::T, x::Dual{T,DT})::Bool where {T,DT}
    a < x.val
end
function Base.:(<)(a::Integer, x::Dual{T,DT})::Bool where {T,DT}
    a < x.val
end

function Base.isinf(x::Dual{T,DT})::Bool where {T,DT}
    isinf(x.val)
end
function Base.isnan(x::Dual{T,DT})::Bool where {T,DT}
    isnan(x.val) || any(isnan.(x.eps))
end

function Base.hash(x::Dual{T,DT}, h::UInt)::UInt where {T,DT}
    hash(0xdccda268, hash(x.val, hash(x.eps, h)))
end
function Base.isequal(x::Dual{T,DT}, y::Dual{T,DT})::Bool where {T,DT}
    isequal(x.val, y.val) && isequal(x.eps, y.eps)
end
function Base.isless(x::Dual{T,DT}, y::Dual{T,DT})::Bool where {T,DT}
    isless(x.val, y.val) && return true
    isless(y.val, x.val) && return false
    isless(x.eps, y.eps)
end



################################################################################

export D
const D = 4



export minkowski
"""
Minkowski metric g_ab
"""
function minkowski(x::SVector{D,T})::SMatrix{D,D,T} where {T}
    @SMatrix T[a==b ? (a==1 ? -1 : 1) : 0 for a in 1:D, b in 1:D]
end



export kerr_schild
"""
Kerr-Schild metric g_ab

Living Reviews in Relativity, Greg Cook, 2000, section 3.3.1
"""
function kerr_schild(xx::SVector{D,T})::SMatrix{D,D,T} where {T}
    M = 1
    a = 0                       # T(0.8)

    t,x,y,z = xx
    @assert !any(isnan, (t, x, y, z))

    # <https://en.wikipedia.org/wiki/Kerr_metric>
    η = @SMatrix T[a==b ? (a==1 ? -1 : 1) : 0 for a in 1:D, b in 1:D]
    ρ = sqrt(x^2 + y^2 + z^2)
    r = sqrt(ρ^2 - a^2)/2 + sqrt(a^2*z^2 + ((ρ^2 - a^2)/2)^2)
    f = 2*M*r^3 / (r^4 + a^2*z^2)
    k = SVector{D,T}(1,
                     (r*x + a*y) / (r^2 + a^2),
                     (r*y - a*x) / (r^2 + a^2),
                     z / r)

    g = @SMatrix T[η[a,b] + f * k[a] * k[b] for a in 1:D, b in 1:D]

    g
end



export dmetric
"""
Derivative of four-metric g_ab,c
"""
function dmetric(metric::Metric, x::SVector{D,T}) where {Metric, T}
    DT = SVector{D,T}
    TDT = Dual{T,DT}
    xdx = SVector{D,TDT}(TDT(x[1], SVector{D,T}(1, 0, 0, 0)),
                         TDT(x[2], SVector{D,T}(0, 1, 0, 0)),
                         TDT(x[3], SVector{D,T}(0, 0, 1, 0)),
                         TDT(x[4], SVector{D,T}(0, 0, 0, 1)))
    gdg = metric(xdx)
    g = @SMatrix T[gdg[a,b].val for a in 1:D, b in 1:D]
    dg = @SArray T[gdg[a,b].eps[c] for a in 1:D, b in 1:D, c in 1:D]
    g, dg
end



export christoffel
"""
Christoffel symbol Γ^a_bc
"""
function christoffel(metric::Metric, x::SVector{D,T}) where {Metric, T}
    g,dg = dmetric(metric, x)
    gu = inv(g)
    Γl = @SArray T[(dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2
                   for a in 1:D, b in 1:D, c in 1:D]
    @SArray T[gu[a,1] * Γl[1,b,c] +
              gu[a,2] * Γl[2,b,c] +
              gu[a,3] * Γl[3,b,c] +
              gu[a,4] * Γl[4,b,c]
              for a in 1:D, b in 1:D, c in 1:D]
end



export Ray
"""
State of a ray: position x, velocity u
"""
struct Ray{T}
    x::SVector{D,T}
    u::SVector{D,T}
end

export r2s, s2r
function r2s(r::Ray{T})::SVector{2D,T} where {T}
    @SVector T[a<=D ? r.x[a] : r.u[a-D] for a in 1:2D]
end
function s2r(s::SVector{2D,T})::Ray{T} where {T}
    x = @SVector T[s[a] for a in 1:D]
    u = @SVector T[s[D+a] for a in 1:D]
    Ray{T}(x, u)
end

export geodesic
"""
RHS of geodesic equation for ray p
"""
function geodesic(r::Ray{T}, metric::Metric, λ::T)::Ray{T} where {Metric, T}
    Γ = christoffel(metric, r.x)
    xdot = r.u
    udot = @SVector T[- sum(@SMatrix T[Γ[a,x,y] * r.u[x] * r.u[y]
                                       for x in 1:D, y in 1:D])
                      for a in 1:D]
    Ray{T}(xdot, udot)
end

function geodesic(s::SVector{2D,T}, metric::Metric,
                  λ::T)::SVector{2D,T} where {Metric, T}
    r2s(geodesic(s2r(s), metric, λ))
end



export Object
abstract type Object{T} end

"""
distance between object and point

The distance does not need to be the geodesic distance; any distance
measure is fine, as long as it is zero on the surface, positive
outside, and negative inside.
"""
function distance(obj::Object, pos::SVector{D,T})::T where{T}
    @error "Called distance on abstract object"
end
function objcolor(obj::Object, pos::SVector{D,T})::SVector{3,T} where {T}
    @error "Called color on abstract object"
end



export Plane
struct Plane{T} <: Object{T}
    # TOOD: add normal
    time::T
end

function distance(pl::Plane{T}, pos::SVector{D,T})::T where{T}
    pos[1] - pl.time
end
function objcolor(pl::Plane{T}, pos::SVector{D,T})::SVector{3,T} where{T}
    SVector{3,T}(0, T(1)/2, 0)
end



export Sphere
struct Sphere{T} <: Object{T}
    pos::SVector{D,T}
    vel::SVector{D,T}
    radius::T
end

function distance(sph::Sphere{T}, pos::SVector{D,T})::T where{T}
    # TODO: Use metric?
    sign(sph.radius) *
        (sum(@SVector T[(pos[a] - sph.pos[a])^2 for a in 2:D]) - sph.radius^2)
end
function objcolor(sph::Sphere{T}, pos::SVector{D,T})::SVector{3,T} where{T}
    x = pos[2] - sph.pos[2]
    y = pos[3] - sph.pos[3]
    z = pos[4] - sph.pos[4]
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = atan(y, x)
    SVector{3,T}(mod(12 * θ / π, 1), mod(12 * ϕ / π, 1), T(1))
end



export min_distance
function min_distance(objs::Vector{Object{T}}, s::SVector{2D,T})::T where {T}
    r = s2r(s)
    dmin = T(Inf)
    for obj in objs
        d = distance(obj, r.x)::T
        dmin = min(dmin, d)
    end
    dmin
end



export Pixel
struct Pixel{T}
    pos::SVector{D,T}         # x^a
    normal::SVector{D,T}      # n^a, past null normal to pixel surface
    rgb::SVector{3,T}
end

export Canvas
struct Canvas{T}
    pixels::Array{Pixel{T}, 2}
end

export make_canvas
function make_canvas(metric::Metric,
                     pos::SVector{D,T},
                     widthx::SVector{D,T}, widthy::SVector{D,T},
                     normal::SVector{D,T},
                     ni::Int, nj::Int) where {Metric, D, T}
    pixels = Array{Pixel{T}}(undef, ni, nj)
    for j in 1:nj, i in 1:ni
        dx = (i-T(1)/2) / ni - T(1)/2
        dy = (j-T(1)/2) / nj - T(1)/2
        x = pos + dx * widthx + dy * widthy
        n = normal + dx * widthx + dy * widthy
        g = metric(x)
        gu = inv(g)
        t = gu * SVector{D,T}(1, 0, 0, 0)
        t2 = t' * g * t
        n2 = n' * g * n
        u = (t / sqrt(-t2) + n / sqrt(n2)) / sqrt(T(2))
        pixels[i,j] = Pixel{T}(x, u, zeros(SVector{3,T}))
    end
    Canvas{T}(pixels)
end



export trace_rays
function trace_rays(metric, objs::Vector{Object{T}},
                    c::Canvas{T})::Canvas{T} where {T}
    tol = eps(T)^(T(3)/4)
    l = length(c.pixels)

    condition(s, λ, integrator) = min_distance(objs, s)
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    function input_func(i::Int)
        p = c.pixels[i]
        x,u = p.pos,p.normal
        r2s(Ray{T}(x, u))
    end
    λ0,λ1 = T(0),T(100)
    prob = ODEProblem(geodesic, input_func(1), (λ0, λ1), metric)
    function prob_func(prob,i,repeat)
        ODEProblem(geodesic, input_func(i), (λ0, λ1), metric)
    end
    function output_func(sol,i)
        λend = sol.t[end]
        s2r(sol[end]), false
    end

    probs = EnsembleProblem(prob,
                            prob_func=prob_func,
                            output_func=output_func)
    sols = solve(probs, Tsit5(), callback=cb, trajectories=length(c.pixels),
                 reltol=tol, abstol=tol)

    pixels = Array{Pixel{T}}(undef, size(c.pixels))
    for i in 1:length(pixels)
        p = c.pixels[i]
        r = sols.u[i]
        x,u = r.x,r.u
        omin = 0
        dmin = T(0.01)
        for (o,obj) in enumerate(objs)
            d = distance(obj, x)
            if d < dmin
                omin = o
                dmin = d
            end
        end
        if omin == 0
            col = SVector{3,T}(1, 0, 0) # did not hit any object
        else
            col = objcolor(objs[omin], x) .* (T(omin) / length(objs))
        end
        pixels[i] = Pixel{T}(p.pos, p.normal, col)
    end

    Canvas{T}(pixels)
end



const outdir = "scenes"

function example1()
    T = Float64

    metric = minkowski
    caelum = Sphere{T}(SVector{D,T}(0, 0, 0, 0), SVector{D,T}(1, 0, 0, 0), -10)
    frustum = Plane{T}(-20)
    sphere = Sphere{T}(SVector{D,T}(0, 0, 0, 0), SVector{D,T}(1, 0, 0, 0),
                       T(1)/2)
    objs = Object{T}[caelum, frustum, sphere]

    ni = 200
    nj = 200
    pos = SVector{D,T}(0, 0, -2, 0)
    widthx = SVector{D,T}(0, 1, 0, 0)
    widthy = SVector{D,T}(0, 0, 0, 1)
    normal = SVector{D,T}(0, 0, 1, 0)
    canvas = make_canvas(metric, pos, widthx, widthy, normal, ni, nj)

    canvas = trace_rays(metric, objs, canvas)

    # scene = colorview(RGB,
    #                   T[mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels]',
    #                   T[mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels]',
    #                   T[mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels]')
    scene = colorview(RGB,
                      T[p.rgb[1] for p in canvas.pixels]',
                      T[p.rgb[2] for p in canvas.pixels]',
                      T[p.rgb[3] for p in canvas.pixels]')

    mkpath(outdir)
    file = joinpath(outdir, "sphere.png")
    rm(file, force=true)
    println("Output file is \"$file\"")
    save(file, scene)
end

function example2()
    T = Float64

    metric = kerr_schild
    caelum = Sphere{T}(SVector{D,T}(0, 0, 0, 0), SVector{D,T}(1, 0, 0, 0), -10)
    frustum = Plane{T}(-20)
    sphere = Sphere{T}(SVector{D,T}(0, 4, 0, 0), SVector{D,T}(1, 0, 0, 0),
                       T(1)/2)
    objs = Object{T}[caelum, frustum, sphere]

    ni = 200
    nj = 200
    pos = SVector{D,T}(0, 4, -2, 0)
    widthx = SVector{D,T}(0, 1, 0, 0)
    widthy = SVector{D,T}(0, 0, 0, 1)
    normal = SVector{D,T}(0, 0, 1, 0)
    canvas = make_canvas(metric, pos, widthx, widthy, normal, ni, nj)

    canvas = trace_rays(metric, objs, canvas)

    # scene = colorview(RGB,
    #                   T[mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels]',
    #                   T[mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels]',
    #                   T[mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels]')
    scene = colorview(RGB,
                      T[p.rgb[1] for p in canvas.pixels]',
                      T[p.rgb[2] for p in canvas.pixels]',
                      T[p.rgb[3] for p in canvas.pixels]')

    mkpath(outdir)
    file = joinpath(outdir, "sphere2.png")
    rm(file, force=true)
    println("Output file is \"$file\"")
    save(file, scene)
end

end
