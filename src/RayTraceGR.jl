module RayTraceGR

using DifferentialEquations
using StaticArrays

export D
const D = 4

export Vec, Mat, Ten3
const Vec{T} = SVector{D,T}
const Mat{T} = SMatrix{D,D,T}
const Ten3{T} = SArray{Tuple{D,D,D}, T}



export metric
# TODO: Try SHermitianCompact as metric type
"""
Four-metric g_ab
"""
function metric(x::Vec{T})::Mat{T} where {T}
    Mat{T}(T[
        a==b ? (a==1 ? -1 : 1) : 0
        for a in 1:D, b in 1:D])
end

export dmetric
"""
Derivative of four-metric g_ab,c
"""
# TODO: Use dual numbers to automate this
function dmetric(x::Vec{T})::Ten3{T} where {T}
    Ten3{T}(T[
        0
        for a in 1:D, b in 1:D, c in 1:D])
end

export christoffel
"""
Christoffel symbol Γ^a_bc
"""
function christoffel(x::Vec{T})::Ten3{T} where {T}
    g = metric(x)
    gu = inv(g)
    dg = dmetric(x)
    Γl = Ten3(T[
        (dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2
        for a in 1:D, b in 1:D, c in 1:D])
    Ten3(T[
        sum(gu[a,x] * Γl[x,b,c] for x in 1:D)
        for a in 1:D, b in 1:D, c in 1:D])
end



export Ray
"""
State of a ray: position x, velocity v
"""
struct Ray{T}
    x::Vec{T}
    v::Vec{T}
end

export r2s, s2r
function r2s(r::Ray{T})::SVector{2*D,T} where {T}
    SVector{2*D,T}(T[r.x; r.v])
end
function s2r(s::SVector{2*D,T})::Ray{T} where {T}
    Ray{T}(s[1:4], s[5:8])
end

export geodesic
"""
RHS of geodesic equation for ray p
"""
function geodesic(r::Ray{T}, par, λ::T)::Ray{T} where {T}
    Γ = christoffel(r.x)
    xdot = r.v
    vdot = - Vec{T}(
        T[sum(Γ[a,x,y] * r.v[x] * r.v[y] for x=1:D, y=1:D)
          for a in 1:D])
    Ray{T}(xdot, vdot)
end

function geodesic(s::SVector{2*D,T}, par, λ::T)::SVector{2*D,T} where {T}
    r2s(geodesic(s2r(s), par, λ))
end

end
