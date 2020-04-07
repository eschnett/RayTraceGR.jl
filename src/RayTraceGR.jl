module RayTraceGR

using OrdinaryDiffEq
using Images
using StaticArrays



struct Dual{T,DT} <: AbstractFloat
    val::T
    eps::DT
end

Base.promote_rule(::Type{Dual{T,DT}}, ::Type{T}) where {T,DT} = Dual{T,DT}
Base.promote_rule(::Type{Dual{T,DT}}, ::Type{Dual{U,DU}}) where {T,DT,U,DU} =
    Dual{promote{T,U}, promote{DT,DU}}

# Cannot have a return type annotation here, as this would lead to an
# infinite recursion
function Base.convert(::Type{Dual{T,DT}}, x::Dual{U,DU}) where {T,DT,U,DU}
    Dual{T,DT}(convert(T, x.val), convert(DT, x.eps))
end
function Base.convert(::Type{Dual{T,DT}}, val::Number)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(convert(T, val), zeros(DT))
end

function Base.zero(::Type{Dual{T,DT}})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(zero(T), zeros(DT))
end
function Base.one(::Type{Dual{T,DT}})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(one(T), zeros(DT))
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
function Base.:-(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val-y.val, x.eps-y.eps)
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
# 1/(a+ϵb) = (a-ϵb)/a^2
function Base.inv(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(inv(x.val), -inv(x.val)^2 .* x.eps)
end
function Base.:/(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val/y.val, (x.eps * y.val - x.val * y.eps) / y.val^2)
end
function Base.:\(x::Dual{T,DT}, y::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val\y.val, x.val^2 \ (- x.val .* y.eps + x.eps .* y.val))
end
function Base.:/(x::Dual{T,DT}, a::T)::Dual{T,DT} where {T,DT}
    Dual{T,DT}(x.val/a, x.eps./a)
end
function Base.:\(a::T, x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(a\x.val, a.\x.eps)
end

function Base.:^(x::Dual{T,DT}, n::Integer)::Dual{T,DT} where {T,DT}
    n == 0 && return one(x)
    Dual{T,DT}(x.val^n, T(n) * x.val^(n-1) .* x.eps)
end

function Base.abs(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(abs(x.val), copysign(one(T), x.val) .* x.eps)
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

function Base.sin(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    Dual{T,DT}(sin(x.val), cos(x.val) .* x.eps)
end

function Base.sqrt(x::Dual{T,DT})::Dual{T,DT} where {T,DT}
    r = sqrt(x.val)
    Dual{T,DT}(r, r/(2*x.val) .* x.eps)
end



################################################################################

export D
const D = 4

export Vec, Mat, Ten3
const Vec{T} = SVector{D,T}
const Mat{T} = SMatrix{D,D,T}
const Ten3{T} = SArray{Tuple{D,D,D}, T}



function minkowski(x::Vec{T})::Mat{T} where {T}
    Mat{T}(T[
        a==b ? (a==1 ? -1 : 1) : 0
        for a in 1:D, b in 1:D])
end

# """
# Kerr-Schild metric
# 
# Living Reviews in Relativity, Greg Cook, 2000, section 3.3.1
# """
# function kerr_schild(xx::Vec{T})::Mat{T} where {T}
#     M = T(1)
#     a = T(0.8)
#     Q = T(0)
# 
#     t,x,y,z = xx
# 
#     R = sqrt(x^2 + y^2 + z^2)
#     cosθ = z / R
#     sinθ = sqrt(x^2 + y^2) / R
# 
#     r = sqrt((R^2 - a^2)/2 + sqrt(a^2*z^2 + ((R^2 - a^2)/2)^2))
#     ρ = sqrt(r^2 + a^2 * cosθ^2)
# 
#     α = 1 / sqrt(1 + (2M*r - Q^2) / ρ^2)
#     βr = α^2 * (2M*r - Q^2) / ρ^2
#     γrr = 1 + (2M*r - Q^2) / ρ^2
#     γrϕ = - (1 + (2M*r - Q^2) / ρ^2) * a * sinθ^2
#     γθθ = ρ^2
#     γϕϕ = (r^2 + a^2 + (2M*r - Q^2) / ρ^2 * a^2 * sinθ^2) * sinθ^2
# 
#     βlr = γrr * βr
# 
#     βlx = βlr * drdx
#     βly = βlr * drdy
#     βlz = βlr * drdz
# 
#     γxx = γrr * drdx * drdx
# 
#     Mat{T}(T[-α^2+β2 βlr 0 0;
#              βlr γrr
#         a==b ? (a==1 ? -1 : 1) : 0
#         for a in 1:D, b in 1:D])
# end
# function kerr_schild(x::Vec{T})::Ten3{T} where {T}
#     Ten3{T}(T[
#         0
#         for a in 1:D, b in 1:D, c in 1:D])
# end



export metric
# TODO: Try SHermitianCompact as metric type
"""
Four-metric g_ab
"""
const metric = minkowski



export dmetric
"""
Derivative of four-metric g_ab,c
"""
# TODO: Use dual numbers to automate this
function dmetric(x::Vec{T})::Tuple{Mat{T}, Ten3{T}} where {T}
    DT = Vec{T}
    TDT = Dual{T,DT}
    xdx = Vec{TDT}(TDT[
        TDT(x[a], DT(T[b==a ? 1 : 0 for b in 1:D]))
        for a in 1:D])
    gdg = metric(xdx)
    g = Mat{T}(T[gdg[a,b].val for a in 1:D, b in 1:D])
    dg = Ten3{T}(T[gdg[a,b].eps[c] for a in 1:D, b in 1:D, c in 1:D])
    @assert g == metric(x)
    (g, dg)
end



export christoffel
"""
Christoffel symbol Γ^a_bc
"""
function christoffel(x::Vec{T})::Ten3{T} where {T}
    g,dg = dmetric(x)
    gu = inv(g)
    Γl = Ten3(T[
        (dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2
        for a in 1:D, b in 1:D, c in 1:D])
    Ten3(T[
        sum(gu[a,x] * Γl[x,b,c] for x in 1:D)
        for a in 1:D, b in 1:D, c in 1:D])
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
    SVector{2D,T}(T[r.x; r.u])
end
function s2r(s::SVector{2D,T})::Ray{T} where {T}
    Ray{T}(s[1:D], s[D+1:2D])
end

export geodesic
"""
RHS of geodesic equation for ray p
"""
function geodesic(r::Ray{T}, par, λ::T)::Ray{T} where {T}
    Γ = christoffel(r.x)
    xdot = r.u
    udot = - Vec{T}(
        T[sum(Γ[a,x,y] * r.u[x] * r.u[y] for x=1:D, y=1:D)
          for a in 1:D])
    Ray{T}(xdot, udot)
end

function geodesic(s::SVector{2D,T}, par, λ::T)::SVector{2D,T} where {T}
    r2s(geodesic(s2r(s), par, λ))
end



export Object
abstract type Object end

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
struct Sphere{T} <: Object
    pos::Vec{T}
    vel::Vec{T}
    radius::T
end

function distance(sph::Sphere{T}, pos::Vec{T})::T where{T}
    # TOOD: use metric
    sum((pos[2:D] - sph.pos[2:D]).^2) - sph.radius^2
end



export Pixel
struct Pixel{T}
    pos::Vec{T}                 # x^a
    normal::Vec{T}              # n^a, spacelike normal to pixel surface
    rgb::SVector{3,T}
end

export trace_ray
function trace_ray(objs::Vector{Object}, p::Pixel{T})::Pixel{T} where {T}
    x = p.pos
    g = metric(x)
    n = p.normal
    # Note: make v null
    # t = [-α, 0]
    # t.g.n == 0
    # u = t + n
    # u.g.u == 0
    v = n + Vec{T}(-1, 0, 0, 0)
    λ0 = T(0)
    λ1 = T(10)
    r = Ray{T}(x, v)
    prob = ODEProblem(geodesic, r2s(r), (λ0, λ1))
    function condition(s, λ, integrator)
        isempty(objs) && return T(1)
        r = s2r(s)::Ray{T}
        minimum(distance(obj, r.x) for obj in objs)
    end
    function affect!(integrator)
        terminate!(integrator)
    end
    cb = ContinuousCallback(condition, affect!)
    tol = eps(T)^(T(3)/4)
    sol = solve(prob, Tsit5(), callback=cb, reltol=tol, abstol=tol)
    λend = sol.t[end]
    rgb = s2r(sol(λend)).x[2:4]
    Pixel{T}(p.pos, p.normal, rgb)
end



export Canvas
struct Canvas{T}
    pixels::Array{Pixel{T}, 2}
end

export trace_rays
function trace_rays(objs::Vector{Object}, c::Canvas{T})::Canvas{T} where {T}
    l = length(c.pixels)
    # TODO: Use EnsembleProblem for parallelism?
    c = Canvas{T}([
    begin
        print("\r$(round(Int, 100*i/l))%")
        trace_ray(objs, p)
    end
    for (i,p) in enumerate(c.pixels)])
    println("\r100%")
    c
end



function main()
    T = Float32

    sph = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(0, 0, 0, 0), 1)
    objs = Object[sph]

    ni = 100
    nj = 100
    function make_pixel(i, j)
        dx = (i-1/2)/ni-1/2
        dy = (j-1/2)/nj-1/2
        x = Vec{T}(0, dx, dy, -2)
        n = Vec{T}(0, dx, dy, 1)
        Pixel{T}(x, n, zeros(SVector{3,T}))
    end
    canvas = Canvas{T}([make_pixel(i,j) for i in 1:ni, j in 1:nj])

    canvas = trace_rays(objs, canvas)

    scene = colorview(RGB,
                      [mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels],
                      [mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels],
                      [mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels])
    # dir = mktempdir()
    dir = "scenes"
    mkpath(dir)
    file = joinpath(dir, "scene.png")
    rm(file, force=true)
    println("Output file is \"$file\"")
    save(file, scene)
end

end
