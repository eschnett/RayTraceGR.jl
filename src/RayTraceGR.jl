module RayTraceGR

using OrdinaryDiffEq
using Images
using LinearAlgebra
using StaticArrays



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
    Dual{T,DT}(r, r/(2*x.val) .* x.eps)
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



export minkowski
"""
Minkowski metric g_ab
"""
function minkowski(x::Vec{T})::Mat{T} where {T}
    Mat{T}(T[
        a==b ? (a==1 ? -1 : 1) : 0
        for a in 1:D, b in 1:D])
end



export kerr_schild
"""
Kerr-Schild metric g_ab

Living Reviews in Relativity, Greg Cook, 2000, section 3.3.1
"""
function kerr_schild(xx::Vec{T})::Mat{T} where {T}
    M = 0                       # 1
    a = 0                       # T(0.8)
    Q = 0

    t,x,y,z = xx

    # To avoid coordinate a singularity on the z axis, all ϕ
    # components are divided by sin θ

    R = sqrt(x^2 + y^2 + z^2)
    θ = atan(sqrt(x^2 + y^2), z)
    cosθ = cos(θ)
    sinθ = sin(θ)
    ϕ = atan(y, x)
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)

    r = sqrt((R^2 - a^2)/2 + sqrt(a^2*z^2 + ((R^2 - a^2)/2)^2))
    ρ = sqrt(r^2 + a^2 * cosθ^2)

    α = 1 / sqrt(1 + (2M*r - Q^2) / ρ^2)
    βr = α^2 * (2M*r - Q^2) / ρ^2
    γrr = 1 + (2M*r - Q^2) / ρ^2
    γrϕ = - (1 + (2M*r - Q^2) / ρ^2) * a * sinθ
    γθθ = ρ^2
    γϕϕ = r^2 + a^2 + (2M*r - Q^2) / ρ^2 * a^2 * sinθ^2

    grr = Mat{T}(T[
        -α^2+βr*γrr*βr   γrr*βr   0     0  ;
        γrr*βr           γrr      0     γrϕ;
        0                0        γθθ   0  ;
        0                γrϕ      0     γϕϕ])

    grr = Mat{T}(T[
        -1   0   0     0  ;
        0    1   0     0  ;
        0    0   R^2   0  ;
        0    0   0     R^2])

    # t = t
    # x = R*sinθ*cosϕ
    # y = R*sinθ*sinϕ
    # z = R*cosθ
    dxdR = Mat{T}(T[
        1   0     0             0      ;
        0   x/R   R*cosθ*cosϕ   -R*sinϕ;
        0   y/R   R*cosθ*sinϕ   R*cosϕ ;
        0   z/R   -R*sinθ       0      ])

    tol = eps(T)^(T(3)/4)
    @assert abs(det(dxdR)) > tol

    dRdx = inv(dxdR)

    Mat{T}(T[
        sum(dRdx[a,x] * dRdx[b,y] * grr[x,y] for x in 1:D, y in 1:D)
        for a in 1:D, b in 1:D])
end



export dmetric
"""
Derivative of four-metric g_ab,c
"""
# TODO: Use dual numbers to automate this
function dmetric(metric, x::Vec{T})::Tuple{Mat{T}, Ten3{T}} where {T}
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
function christoffel(metric, x::Vec{T})::Ten3{T} where {T}
    g,dg = dmetric(metric, x)
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
function geodesic(r::Ray{T}, metric, λ::T)::Ray{T} where {T}
    Γ = christoffel(metric, r.x)
    xdot = r.u
    udot = - Vec{T}(T[
        sum(Γ[a,x,y] * r.u[x] * r.u[y] for x=1:D, y=1:D)
        for a in 1:D])
    Ray{T}(xdot, udot)
end

function geodesic(s::SVector{2D,T}, metric, λ::T)::SVector{2D,T} where {T}
    r2s(geodesic( s2r(s), metric, λ))
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
function trace_ray(metric, objs::Vector{Object},
                   p::Pixel{T})::Pixel{T} where {T}
    # tol = eps(T)^(T(3)/4)
    tol = eps(T)^(T(1)/2)

    x = p.pos
    g = metric(x)
    n = p.normal
    @show "trace_ray" x

    t = Vec{T}(-1, 0, 0, 0)
    nn = n' * g * n
    tt = t' * g * t
    nt = n' * g * t
    α = sqrt((nt/tt)^2 - nn/tt) - nt/tt
    v = n + α * t
    vv = v' * g * v
    @assert abs(vv) <= tol

    λ0 = T(0)
    λ1 = T(10)
    r = Ray{T}(x, v)
    @show "trace_ray" r
    prob = ODEProblem(geodesic, r2s(r), (λ0, λ1), metric)
    function condition(s, λ, integrator)
        isempty(objs) && return T(1)
        r = s2r(s)::Ray{T}
        minimum(distance(obj, r.x) for obj in objs)
    end
    function affect!(integrator)
        terminate!(integrator)
    end
    cb = ContinuousCallback(condition, affect!)
    @show "trace_ray.before"
    sol = solve(prob, Tsit5(), callback=cb, reltol=tol, abstol=tol)
    @show "trace_ray.after"
    λend = sol.t[end]
    @show "trace_ray" λend
    rgb = s2r(sol(λend)).x[2:4]
    @show "trace_ray" rgb
    Pixel{T}(p.pos, p.normal, rgb)
end



export Canvas
struct Canvas{T}
    pixels::Array{Pixel{T}, 2}
end

export trace_rays
function trace_rays(metric, objs::Vector{Object},
                    c::Canvas{T})::Canvas{T} where {T}
    l = length(c.pixels)
    # TODO: Use EnsembleProblem for parallelism?
    c = Canvas{T}([
    begin
        print("\r$(round(Int, 100*i/l))%")
        trace_ray(metric, objs, p)
    end
    for (i,p) in enumerate(c.pixels)])
    println("\r100%")
    c
end



const outdir = "scenes"

function example1()
    T = Float32

    sph = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(0, 0, 0, 0), 1)
    objs = Object[sph]

    ni = 100
    nj = 100
    pos = Vec{T}(0, 0, 0, -2)
    widthx = Vec{T}(0, 1, 0, 0)
    widthy = Vec{T}(0, 0, 1, 0)
    normal = Vec{T}(0, 0, 0, 1)
    function make_pixel(i, j)
        dx = (i-1/2) / ni - 1/2
        dy = (j-1/2) / nj - 1/2
        x = pos + dx * widthx + dy * widthy
        n = normal + dx * widthx + dy * widthy
        Pixel{T}(x, n, zeros(SVector{3,T}))
    end
    canvas = Canvas{T}([make_pixel(i,j) for i in 1:ni, j in 1:nj])

    canvas = trace_rays(minkowski, objs, canvas)

    scene = colorview(RGB,
                      [mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels],
                      [mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels],
                      [mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels])
    mkpath(outdir)
    file = joinpath(outdir, "sphere.png")
    rm(file, force=true)
    println("Output file is \"$file\"")
    save(file, scene)
    nothing
end

function example2()
    T = Float32

    sph = Sphere{T}(Vec{T}(0, 4, 0, 0), Vec{T}(0, 0, 0, 0), 1)
    objs = Object[sph]

    ni = 100
    nj = 100
    pos = Vec{T}(0, 2, 0, -4)
    widthx = Vec{T}(0, 1, 0, 0)
    widthy = Vec{T}(0, 0, 1, 0)
    normal = Vec{T}(0, 0, 0, 1)
    function make_pixel(i, j)
        dx = (i-1/2) / ni - 1/2
        dy = (j-1/2) / nj - 1/2
        x = pos + dx * widthx + dy * widthy
        n = normal + dx * widthx + dy * widthy
        Pixel{T}(x, n, zeros(SVector{3,T}))
    end
    canvas = Canvas{T}([make_pixel(i,j) for i in 1:ni, j in 1:nj])

    canvas = trace_rays(kerr_schild, objs, canvas)

    scene = colorview(RGB,
                      [mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels],
                      [mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels],
                      [mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels])
    mkpath(outdir)
    file = joinpath(outdir, "sphere2.png")
    rm(file, force=true)
    println("Output file is \"$file\"")
    save(file, scene)
    nothing
end

end
