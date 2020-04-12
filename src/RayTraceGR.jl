module RayTraceGR

using OrdinaryDiffEq
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

export Vec, Mat, Ten3
const Vec{T} = SVector{D,T}
const Mat{T} = SMatrix{D,D,T}
const Ten3{T} = SArray{Tuple{D,D,D}, T}



export ssum, ssum2, ssum3
function ssum(::Type{T}, f::F) where {T, F}
    s = T(0)
    for i in 1:D
        s += f(i)::T
    end
    s
end
function ssum2(::Type{T}, f::F) where {T, F}
    s = T(0)
    for i in 1:D, j in 1:D
        s += f(i, j)::T
    end
    s
end
function ssum3(::Type{T}, f::F) where {T, F}
    s = T(0)
    for i in 1:D, j in 1:D, k in 1:D
        s += f(i, j, k)::T
    end
    s
end



################################################################################



export minkowski
"""
Minkowski metric g_ab
"""
function minkowski(x::Vec{T})::Mat{T} where {T}
    @SMatrix T[a==b ? (a==1 ? -1 : 1) : 0 for a in 1:D, b in 1:D]
end



export kerr_schild
"""
Kerr-Schild metric g_ab

Living Reviews in Relativity, Greg Cook, 2000, section 3.3.1
"""
function kerr_schild(xx::Vec{T})::Mat{T} where {T}
    M = 1
    a = 0                       # T(0.8)
    Q = 0

    t,x,y,z = xx
    @assert !any(isnan, (t, x, y, z))

    # <https://en.wikipedia.org/wiki/Kerr_metric>
    η = @SMatrix T[-1 0 0 0;
                   0  1 0 0;
                   0  0 1 0;
                   0  0 0 1]
    ρ = sqrt(x^2 + y^2 + z^2)
    r = sqrt(ρ^2 - a^2)/2 + sqrt(a^2*z^2 + ((ρ^2 - a^2)/2)^2)
    f = 2*M*r^3 / (r^4 + a^2*z^2)
    k = @SVector T[1,
                   (r*x + a*y) / (r^2 + a^2),
                   (r*y - a*x) / (r^2 + a^2),
                   z / r]

    g = @SMatrix T[η[a,b] + f * k[a] * k[b] for a in 1:D, b in 1:D]

    g
end



export dmetric
"""
Derivative of four-metric g_ab,c
"""
# TODO: Use dual numbers to automate this
function dmetric(metric::Metric,
                 x::Vec{T})::Tuple{Mat{T}, Ten3{T}} where {Metric, T}
    DT = Vec{T}
    TDT = Dual{T,DT}
    # xdx = @SVector TDT[
    #     TDT(x[a], @SVector T[b==a ? 1 : 0 for b in 1:D])
    #     for a in 1:D]
    xdx = Vec{TDT}(TDT(x[1], Vec{T}(1, 0, 0, 0)),
                   TDT(x[2], Vec{T}(0, 1, 0, 0)),
                   TDT(x[3], Vec{T}(0, 0, 1, 0)),
                   TDT(x[4], Vec{T}(0, 0, 0, 1)))
    gdg = metric(xdx)
    g = @SMatrix T[gdg[a,b].val for a in 1:D, b in 1:D]
    dg = @SArray T[gdg[a,b].eps[c] for a in 1:D, b in 1:D, c in 1:D]
    (g, dg)
end



export christoffel
"""
Christoffel symbol Γ^a_bc
"""
function christoffel(metric::Metric, x::Vec{T})::Ten3{T} where {Metric, T}
    g,dg = dmetric(metric, x)
    gu = inv(g)
    Γl = @SArray T[
        (dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2
        for a in 1:D, b in 1:D, c in 1:D]
    @SArray T[ssum(T, x -> gu[a,x] * Γl[x,b,c])
              for a in 1:D, b in 1:D, c in 1:D]
end



export Ray
"""
State of a ray: position x, velocity u
"""
struct Ray{T}
    x::Vec{T}
    u::Vec{T}
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
    udot = @SVector T[- ssum2(T, (x,y) -> Γ[a,x,y] * r.u[x] * r.u[y])
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
function distance(obj::Object, pos::Vec)
    @error "Called distance on abstract object"
end



export Sphere
struct Sphere{T} <: Object{T}
    pos::Vec{T}
    vel::Vec{T}
    radius::T
end

function distance(sph::Sphere{T}, pos::Vec{T})::T where{T}
    # TODO: Use metric?
    ssum(T, a -> (pos[a] - sph.pos[a])^2) - sph.radius^2
end



export Pixel
struct Pixel{T}
    pos::Vec{T}                 # x^a
    normal::Vec{T}              # n^a, past null normal to pixel surface
    rgb::SVector{3,T}
end

export trace_ray
function trace_ray(metric, objs::Vector{Object{T}},
                   p::Pixel{T})::Pixel{T} where {T}
    tol = eps(T)^(T(3)/4)

    x,u = p.pos,p.normal
    g = metric(x)
    u2 = u' * g * u
    #@show "trace_ray" x u u2
    @assert abs(u2) <= tol

    λ0 = T(0)
    λ1 = T(10)
    r = Ray{T}(x, u)
    #@show "trace_ray" r
    prob = ODEProblem(geodesic, r2s(r), (λ0, λ1), metric)
    function condition(s::SVector{2D,T}, λ, integrator)
        r = s2r(s)::Ray{T}
        dmin = T(Inf)
        for obj in objs
            d = distance(obj, r.x)::T
            dmin = min(dmin, d)
        end
        dmin
    end
    function affect!(integrator)
        terminate!(integrator)
    end
    cb = ContinuousCallback(condition, affect!)
    #@show "trace_ray.before"
    sol = solve(prob, Tsit5(), callback=cb, reltol=tol, abstol=tol)
    #@show "trace_ray.after"
    λend = sol.t[end]
    #@show λend
    rend = s2r(sol(λend))
    #@show rend
    x,u = rend.x,rend.u
    g = metric(x)
    u2 = u' * g * u
    #@show x u u2
    rgb = @view rend.x[2:4]
    #@show rgb
    Pixel{T}(p.pos, p.normal, rgb)
end



export Canvas
struct Canvas{T}
    pixels::Array{Pixel{T}, 2}
end

export trace_rays
function trace_rays(metric, objs::Vector{Object{T}},
                    c::Canvas{T})::Canvas{T} where {T}
    l = length(c.pixels)
    # TODO: Use EnsembleProblem for parallelism?
    c = Canvas{T}(Pixel{T}[
        (
    print("\r$(round(Int, 100*(i-1)/l))%");
    trace_ray(metric, objs, p)
    )
        for (i,p) in enumerate(c.pixels)])
    println("\r100%")
    c
end



const outdir = "scenes"

function example1()
    T = Float32

    metric = minkowski
    sph = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(0, 0, 0, 0), 1)
    objs = Object{T}[sph]

    ni = 100
    nj = 100
    pos = Vec{T}(0, 0, 0, 2)
    widthx = Vec{T}(0, 1, 0, 0)
    widthy = Vec{T}(0, 0, 1, 0)
    normal = Vec{T}(0, 0, 0, -1)
    function make_pixel(i, j)
        dx = (i-1/2) / ni - 1/2
        dy = (j-1/2) / nj - 1/2
        x = pos + dx * widthx + dy * widthy
        n = normal + dx * widthx + dy * widthy
        g = metric(x)
        gu = inv(g)
        t = gu * Vec{T}(1, 0, 0, 0)
        t2 = t' * g * t
        n2 = n' * g * n
        u = (t / sqrt(-t2) + n / sqrt(n2)) / sqrt(T(2))
        Pixel{T}(x, u, zeros(SVector{3,T}))
    end
    canvas = Canvas{T}(Pixel{T}[make_pixel(i,j) for i in 1:ni, j in 1:nj])

    canvas = trace_rays(metric, objs, canvas)

    scene = colorview(RGB,
                      T[mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels],
                      T[mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels],
                      T[mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels])
    mkpath(outdir)
    file = joinpath(outdir, "sphere.png")
    rm(file, force=true)
    println("Output file is \"$file\"")
    save(file, scene)
    nothing
end

function example2()
    T = Float32

    metric = kerr_schild
    sph = Sphere{T}(Vec{T}(0, 4, 0, 0), Vec{T}(1, 0, 0, 0), 1)
    objs = Object{T}[sph]

    ni = 100
    nj = 100
    pos = Vec{T}(0, 20, 0, 4)
    widthx = Vec{T}(0, 1, 0, 0)
    widthy = Vec{T}(0, 0, 1, 0)
    normal = Vec{T}(0, 0, 0, -1)
    function make_pixel(i, j)
        dx = (i-1/2) / ni - 1/2
        dy = (j-1/2) / nj - 1/2
        x = pos + dx * widthx + dy * widthy
        n = normal + dx * widthx + dy * widthy
        g = metric(x)
        gu = inv(g)
        t = gu * Vec{T}(1, 0, 0, 0)
        t2 = t' * g * t
        n2 = n' * g * n
        u = (t / sqrt(-t2) + n / sqrt(n2)) / sqrt(T(2))
        Pixel{T}(x, u, zeros(SVector{3,T}))
    end
    canvas = Canvas{T}(Pixel{T}[make_pixel(i,j) for i in 1:ni, j in 1:nj])

    canvas = trace_rays(metric, objs, canvas)

    scene = colorview(RGB,
                      T[mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels],
                      T[mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels],
                      T[mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels])
    mkpath(outdir)
    file = joinpath(outdir, "sphere2.png")
    rm(file, force=true)
    println("Output file is \"$file\"")
    save(file, scene)
    nothing
end

end
