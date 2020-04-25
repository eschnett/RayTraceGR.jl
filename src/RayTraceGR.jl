module RayTraceGR

using ComputedFieldTypes
# using Distributed
using Grassmann
# using Images
using LinearAlgebra
# using OrdinaryDiffEq
# using SharedArrays
using StaticArrays



export bitsign
bitsign(b::Bool) = b ? -1 : 1



# TODO: Find an existing implementation
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

# function Base.sum(::Type{T}, f::F, r1::R1) where {T, F, R1}
#     s = zero(T)
#     for i in r1
#         s += f(i)::T
#     end
#     s
# end
# function Base.sum(::Type{T}, f::F, r1::R1, r2::R2) where {T, F, R1, R2}
#     s = zero(T)
#     for i in r1, j in r2
#         s += f(i, j)::T
#     end
#     s
# end
# function Base.sum(::Type{T}, f::F, r1::R1, r2::R2, r3::R3
#                   ) where {T, F, R1, R2, R3}
#     s = zero(T)
#     for i in r1, j in r2, k in r3
#         s += f(i, j, k)::T
#     end
#     s
# end



################################################################################

export S, D
const S = S"++++"               # signature
const D = ndims(S)              # dimensions

export V, v
@basis S                        # define convenience names



function Base.rand(::Type{<:Chain{V,1,T}}) where {V, T}
    D = ndims(V)
    Chain{V,1}(SVector{D,T}((rand(T) for i in 1:ndims(V))...))::Chain{V,1,<:T}
end



export detM
function detM(A::Chain{V,1, <:Chain{V,1,T}}) where {V, T}
    # Determinant via SMatrix
    D = ndims(V)
    A1 = SMatrix{D,D,T}((A.v[j].v[i] for i in 1:D, j in 1:D)...)
    det(A1)::T
end

export invM
function invM(A::Chain{V,1, <:Chain{V,1,T}}) where {V, T}
    # Invert via SMatrix
    D = ndims(V)
    CT = fulltype(Chain{V,1,T})
    A1 = SMatrix{D,D,T}((A.v[j].v[i] for i in 1:D, j in 1:D)...)
    B1 = inv(A1)
    Chain{V,1}(SVector{D,CT}(
        (Chain{V,1}(SVector{D,T}((B1[i,j] for i in 1:D)...))
         for j in 1:D)...))::Chain{V,1, <:Chain{V,1,T}}
end



# # vector
# export Vec, fullVec, makeVec
# const Vec{T} = Chain{V, 1, T} where {T}
# fullVec(::Type{T}) where {T} = fulltype(Vec{T})
# makeVec(::Type{T}, x::TensorAlgebra) where {T} = Chain{T}(Chain(x))::Vec{T}
# makeVec(::Type{T}, f::F) where {T, F} =
#     sum(Vec{T}, i -> Chain(T(f(i)), v(i))::Vec{T}, 1:D)::Vec{T}
# 
# Base.rand(::Type{<:Vec{T}}) where {T} = makeVec(T, d->rand(T))::Vec{T}
# 
# 
# 
# # bivector
# export Vec2, fullVec2, makeVec2
# const Vec2{T} = Chain{V, 2, T} where {T}
# fullVec2(::Type{T}) where {T} = fulltype(Vec2{T})
# makeVec2(::Type{T}, x::TensorAlgebra) where {T} = Chain{T}(Chain(x))::Vec2{T}
# makeVec2(::Type{T}, f::F) where {T, F} =
#     sum(Vec{T},
#         i -> sum(Vec{T}, j -> Chain(T(f(i,j)), v(i,j))::Vec{T}, i+1:D),
#         1:D)::Vec2{T}
# 
# Base.rand(::Type{<:Vec2{T}}) where {T} = makeVec2(T, d->rand(T))::Vec2{T}
# 
# 
# 
# # vector-valued linear function of a vector
# export Mat, fullMat, makeMat
# @computed struct Mat{T}
#     elts::NTuple{D, fullVec(T)}
# end
# Base.zero(::Type{<:Mat{T}}) where {T} = Mat{T}(ntuple(i -> zero(Vec{T}), D))
# 
# fullMat(::Type{T}) where {T} = fulltype(Mat{T})
# makeMat(::Type{T}, xs::F) where {T, F} = Mat{T}(ntuple(d->makeVec(T,xs(d)), D))
# 
# Base.rand(::Type{<:Mat{T}}) where {T} = makeMat(T, d->rand(Vec{T}))::Mat{T}
# 
# function (A::Mat{T})(x::Vec{T}) where {T}
#     sum(Chain{V,1,T}, d -> x.v[d] * A.elts[d], 1:D)::Vec{T}
# end
# 
# function LinearAlgebra.det(A::Mat{T}) where {T}
#     detA = one(MultiVector{V,T})
#     for d in 1:D
#         detA = detA ∧ A(makeVec(T, v(d)))
#     end
#     detA.v[end]::T
# end
# 
# function Base.inv(A::Mat{T}) where {T}
#     A1 = SMatrix{D,D}(T[A.elts[j].v[i] for i in 1:D, j in 1:D])
#     A1inv = inv(A1)
#     Ainv = makeMat(T, j->makeVec(T, i->A1inv[i,j]))
#     Ainv::Mat{T}
# end
# 
# 
# 
# # bivector-valued linear function of a vector
# export Mat3, fullMat3, makeMat3
# @computed struct Mat3{T}
#     elts::NTuple{D, fullVec(T)}
# end
# Base.zero(::Type{<:Mat3{T}}) where {T} = Mat3{T}(ntuple(i -> zero(Vec{T}), D))
# 
# fullMat3(::Type{T}) where {T} = fulltype(Mat3{T})
# makeMat3(::Type{T}, xs::Tuple) where {T} = Mat3{T}(map(x->makeVec(T,x), xs))
# 
# function Base.rand(::Type{<:Mat3{T}}) where {T}
#     makeMat3(T, ntuple(d->rand(Vec{T})))::Mat3{T}
# end
# 
# function (A::Mat3{T})(x::Vec{T}) where {T}
#     sum(Chain{V,1,T}, d -> x.v[d] * A.elts[d], 1:D)::Vec{T}
# end



################################################################################



export minkowski
"""
Minkowski metric g_ab
"""
function minkowski(x::Chain{V,1,T}) where {V, T}
    D = ndims(V)
    CT = fulltype(Chain{V,1,T})
    Chain{V,1}(SVector{D,CT}((Chain(T(bitsign(i==1)) * v(i)) for i in 1:D)...))
end



#TODO export kerr_schild
#TODO """
#TODO Kerr-Schild metric g_ab
#TODO 
#TODO Living Reviews in Relativity, Greg Cook, 2000, section 3.3.1
#TODO """
#TODO function kerr_schild(xx::Vec{T})::Mat{T} where {T}
#TODO     M = 1
#TODO     a = 0                       # T(0.8)
#TODO 
#TODO     t,x,y,z = xx
#TODO     @assert !any(isnan, (t, x, y, z))
#TODO 
#TODO     # <https://en.wikipedia.org/wiki/Kerr_metric>
#TODO     η = smat(T, (a,b) -> a==b ? (a==1 ? -1 : 1) : 0)
#TODO     ρ = sqrt(x^2 + y^2 + z^2)
#TODO     r = sqrt(ρ^2 - a^2)/2 + sqrt(a^2*z^2 + ((ρ^2 - a^2)/2)^2)
#TODO     f = 2*M*r^3 / (r^4 + a^2*z^2)
#TODO     k = @SVector T[1,
#TODO                    (r*x + a*y) / (r^2 + a^2),
#TODO                    (r*y - a*x) / (r^2 + a^2),
#TODO                    z / r]
#TODO 
#TODO     g = smat(T, (a,b) -> η[a,b] + f * k[a] * k[b])
#TODO 
#TODO     g
#TODO end



# export dmetric
# """
# Derivative of four-metric g_ab,c
# """
# # TODO: Use dual numbers to automate this
# function dmetric(metric::Metric,
#                  x::Vec{T})::Tuple{Mat{T}, Ten3{T}} where {Metric, T}
#     DT = Vec{T}
#     TDT = Dual{T,DT}
#     # xdx = @SVector TDT[
#     #     TDT(x[a], @SVector T[b==a ? 1 : 0 for b in 1:D])
#     #     for a in 1:D]
#     xdx = Vec{TDT}(TDT(x[1], Vec{T}(1, 0, 0, 0)),
#                    TDT(x[2], Vec{T}(0, 1, 0, 0)),
#                    TDT(x[3], Vec{T}(0, 0, 1, 0)),
#                    TDT(x[4], Vec{T}(0, 0, 0, 1)))
#     gdg = metric(xdx)
#     g = smat(T, (a,b) -> gdg[a,b].val)
#     dg = sten3(T, (a,b,c) -> gdg[a,b].eps[c])
#     (g, dg)
# end



#TODO export christoffel
#TODO """
#TODO Christoffel symbol Γ^a_bc
#TODO """
#TODO function christoffel(metric::Metric, x::Vec{T})::Ten3{T} where {Metric, T}
#TODO     g,dg = dmetric(metric, x)
#TODO     gu = inv(g)
#TODO     Γl = sten3(T, (a,b,c) -> (dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2)
#TODO     sten3(T, (a,b,c) -> ssum(T, x -> gu[a,x] * Γl[x,b,c]))
#TODO end
#TODO 
#TODO 
#TODO 
#TODO export Ray
#TODO """
#TODO State of a ray: position x, velocity u
#TODO """
#TODO struct Ray{T}
#TODO     x::Vec{T}
#TODO     u::Vec{T}
#TODO end
#TODO 
#TODO export r2s, s2r
#TODO function r2s(r::Ray{T})::SVector{2D,T} where {T}
#TODO     @SVector T[a<=D ? r.x[a] : r.u[a-D] for a in 1:2D]
#TODO end
#TODO function s2r(s::SVector{2D,T})::Ray{T} where {T}
#TODO     x = @SVector T[s[a] for a in 1:D]
#TODO     u = @SVector T[s[D+a] for a in 1:D]
#TODO     Ray{T}(x, u)
#TODO end
#TODO 
#TODO export geodesic
#TODO """
#TODO RHS of geodesic equation for ray p
#TODO """
#TODO function geodesic(r::Ray{T}, metric::Metric, λ::T)::Ray{T} where {Metric, T}
#TODO     Γ = christoffel(metric, r.x)
#TODO     xdot = r.u
#TODO     udot = svec(T, a -> - ssum2(T, (x,y) -> Γ[a,x,y] * r.u[x] * r.u[y]))
#TODO     Ray{T}(xdot, udot)
#TODO end
#TODO 
#TODO function geodesic(s::SVector{2D,T}, metric::Metric,
#TODO                   λ::T)::SVector{2D,T} where {Metric, T}
#TODO     r2s(geodesic(s2r(s), metric, λ))
#TODO end
#TODO 
#TODO 
#TODO 
#TODO export Object
#TODO abstract type Object{T} end
#TODO 
#TODO """
#TODO distance between object and point
#TODO 
#TODO The distance does not need to be the geodesic distance; any distance
#TODO measure is fine, as long as it is zero on the surface, positive
#TODO outside, and negative inside.
#TODO """
#TODO function distance(obj::Object, pos::Vec)
#TODO     @error "Called distance on abstract object"
#TODO end
#TODO function objcolor(obj::Object, pos::Vec)
#TODO     @error "Called color on abstract object"
#TODO end
#TODO 
#TODO 
#TODO 
#TODO export Plane
#TODO struct Plane{T} <: Object{T}
#TODO     # TOOD: add normal
#TODO     time::T
#TODO end
#TODO 
#TODO function distance(pl::Plane{T}, pos::Vec{T})::T where{T}
#TODO     pos[1] - pl.time
#TODO end
#TODO function objcolor(pl::Plane{T}, pos::Vec{T})::SVector{3,T} where{T}
#TODO     SVector{3,T}(0, T(1)/2, 0)
#TODO end
#TODO 
#TODO 
#TODO 
#TODO export Sphere
#TODO struct Sphere{T} <: Object{T}
#TODO     pos::Vec{T}
#TODO     vel::Vec{T}
#TODO     radius::T
#TODO end
#TODO 
#TODO function distance(sph::Sphere{T}, pos::Vec{T})::T where{T}
#TODO     # TODO: Use metric?
#TODO     sign(sph.radius) *
#TODO         (ssum(T, a -> (pos[a] - sph.pos[a])^2, 2:D) - sph.radius^2)
#TODO end
#TODO function objcolor(sph::Sphere{T}, pos::Vec{T})::SVector{3,T} where{T}
#TODO     x = pos[2] - sph.pos[2]
#TODO     y = pos[3] - sph.pos[3]
#TODO     z = pos[4] - sph.pos[4]
#TODO     r = sqrt(x^2 + y^2 + z^2)
#TODO     θ = acos(z / r)
#TODO     ϕ = atan(y, x)
#TODO     SVector{3,T}(mod(12 * θ / π, 1), mod(12 * ϕ / π, 1), T(1))
#TODO end
#TODO 
#TODO 
#TODO 
#TODO export Pixel
#TODO struct Pixel{T}
#TODO     pos::Vec{T}                 # x^a
#TODO     normal::Vec{T}              # n^a, past null normal to pixel surface
#TODO     rgb::SVector{3,T}
#TODO end
#TODO 
#TODO export trace_ray
#TODO function trace_ray(metric, objs::Vector{Object{T}},
#TODO                    p::Pixel{T})::Pixel{T} where {T}
#TODO     tol = eps(T)^(T(3)/4)
#TODO 
#TODO     x,u = p.pos,p.normal
#TODO     g = metric(x)
#TODO     u2 = u' * g * u
#TODO     @assert abs(u2) <= tol
#TODO 
#TODO     λ0 = T(0)
#TODO     λ1 = T(100)
#TODO     r = Ray{T}(x, u)
#TODO     prob = ODEProblem(geodesic, r2s(r), (λ0, λ1), metric)
#TODO     function condition(s::SVector{2D,T}, λ, integrator)
#TODO         r = s2r(s)::Ray{T}
#TODO         dmin = T(Inf)
#TODO         for obj in objs
#TODO             d = distance(obj, r.x)::T
#TODO             dmin = min(dmin, d)
#TODO         end
#TODO         dmin
#TODO     end
#TODO     function affect!(integrator)
#TODO         terminate!(integrator)
#TODO     end
#TODO     cb = ContinuousCallback(condition, affect!)
#TODO     sol = solve(prob, Tsit5(), callback=cb, reltol=tol, abstol=tol)
#TODO     λend = sol.t[end]
#TODO     rend = s2r(sol(λend))
#TODO     x,u = rend.x,rend.u
#TODO 
#TODO     # g = metric(x)
#TODO     # u2 = u' * g * u
#TODO     # rgb = @view rend.x[2:4]
#TODO     # Pixel{T}(p.pos, p.normal, rgb)
#TODO 
#TODO     imin = 0
#TODO     dmin = T(0.01)
#TODO     for (i,obj) in enumerate(objs)
#TODO         d = distance(obj, x)
#TODO         if d < dmin
#TODO             imin = i
#TODO             dmin = d
#TODO         end
#TODO     end
#TODO     if imin == 0
#TODO         col = SVector{3,T}(1, 0, 0) # did not hit any object
#TODO     else
#TODO         col = objcolor(objs[imin], x) .* (T(imin) / length(objs))
#TODO     end
#TODO     Pixel{T}(p.pos, p.normal, col)
#TODO end
#TODO 
#TODO 
#TODO 
#TODO export Canvas
#TODO struct Canvas{T}
#TODO     pixels::Array{Pixel{T}, 2}
#TODO end
#TODO 
#TODO export trace_rays
#TODO function trace_rays(metric, objs::Vector{Object{T}},
#TODO                     c::Canvas{T})::Canvas{T} where {T}
#TODO     l = length(c.pixels)
#TODO 
#TODO     # Serial
#TODO     # c = Canvas{T}(Pixel{T}[
#TODO     #     (
#TODO     # print("\r$(round(Int, 100*(i-1)/l))%");
#TODO     # trace_ray(metric, objs, p)
#TODO     # )
#TODO     #     for (i,p) in enumerate(c.pixels)])
#TODO     # println("\r100%")
#TODO 
#TODO     # Distributed
#TODO     # ps = SharedArray{Pixel{T}}(l)
#TODO     # @sync @distributed for i in 1:l
#TODO     #     ps[i] = trace_ray(metric, objs, c.pixels[i])
#TODO     # end
#TODO     # c = Canvas{T}(Array{Pixel{T}, 2}(undef, size(c.pixels)))
#TODO     # for i in 1:l
#TODO     #     c.pixels[i] = ps[i]
#TODO     # end
#TODO 
#TODO     # Multi-threaded per pixel
#TODO     # ps = Array{Pixel{T}, 2}(undef, size(c.pixels))
#TODO     # Threads.@threads for i in 1:l
#TODO     #     ps[i] = trace_ray(metric, objs, c.pixels[i])
#TODO     # end
#TODO     # c = Canvas{T}(ps)
#TODO 
#TODO     # Multi-threaded per scanline
#TODO     ni,nj = size(c.pixels)
#TODO     ps = Array{Pixel{T}, 2}(undef, ni, nj)
#TODO     Threads.@threads for j in 1:nj
#TODO         for i in 1:ni
#TODO             ps[i,j] = trace_ray(metric, objs, c.pixels[i,j])
#TODO         end
#TODO     end
#TODO     c = Canvas{T}(ps)
#TODO 
#TODO     c
#TODO end
#TODO 
#TODO 
#TODO 
#TODO const outdir = "scenes"
#TODO 
#TODO function example1()
#TODO     T = Float64
#TODO 
#TODO     metric = minkowski
#TODO     caelum = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(1, 0, 0, 0), -10)
#TODO     frustum = Plane{T}(-20)
#TODO     sphere = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(1, 0, 0, 0), T(1)/2)
#TODO     objs = Object{T}[caelum, frustum, sphere]
#TODO 
#TODO     ni = 200
#TODO     nj = 200
#TODO     pos = Vec{T}(0, 0, -2, 0)
#TODO     widthx = Vec{T}(0, 1, 0, 0)
#TODO     widthy = Vec{T}(0, 0, 0, 1)
#TODO     normal = Vec{T}(0, 0, 1, 0)
#TODO     function make_pixel(i, j)
#TODO         dx = (i-1/2) / ni - 1/2
#TODO         dy = (j-1/2) / nj - 1/2
#TODO         x = pos + dx * widthx + dy * widthy
#TODO         n = normal + dx * widthx + dy * widthy
#TODO         g = metric(x)
#TODO         gu = inv(g)
#TODO         t = gu * Vec{T}(1, 0, 0, 0)
#TODO         t2 = t' * g * t
#TODO         n2 = n' * g * n
#TODO         u = (t / sqrt(-t2) + n / sqrt(n2)) / sqrt(T(2))
#TODO         Pixel{T}(x, u, zeros(SVector{3,T}))
#TODO     end
#TODO     canvas = Canvas{T}(Pixel{T}[make_pixel(i,j) for i in 1:ni, j in 1:nj])
#TODO 
#TODO     canvas = trace_rays(metric, objs, canvas)
#TODO 
#TODO     # scene = colorview(RGB,
#TODO     #                   T[mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels]',
#TODO     #                   T[mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels]',
#TODO     #                   T[mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels]')
#TODO     scene = colorview(RGB,
#TODO                       T[p.rgb[1] for p in canvas.pixels]',
#TODO                       T[p.rgb[2] for p in canvas.pixels]',
#TODO                       T[p.rgb[3] for p in canvas.pixels]')
#TODO 
#TODO     mkpath(outdir)
#TODO     file = joinpath(outdir, "sphere.png")
#TODO     rm(file, force=true)
#TODO     println("Output file is \"$file\"")
#TODO     save(file, scene)
#TODO     nothing
#TODO end
#TODO 
#TODO function example2()
#TODO     T = Float64
#TODO 
#TODO     metric = kerr_schild
#TODO     caelum = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(1, 0, 0, 0), -10)
#TODO     frustum = Plane{T}(-20)
#TODO     sphere = Sphere{T}(Vec{T}(0, 4, 0, 0), Vec{T}(1, 0, 0, 0), T(1)/2)
#TODO     objs = Object{T}[caelum, frustum, sphere]
#TODO 
#TODO     ni = 200
#TODO     nj = 200
#TODO     # TODO: make screen infinitesimally small, pixels only differ by
#TODO     # their normal direction
#TODO     pos = Vec{T}(0, 4, -2, 0)
#TODO     widthx = Vec{T}(0, 1, 0, 0)
#TODO     widthy = Vec{T}(0, 0, 0, 1)
#TODO     normal = Vec{T}(0, 0, 1, 0)
#TODO     function make_pixel(i, j)
#TODO         dx = (i-1/2) / ni - 1/2
#TODO         dy = (j-1/2) / nj - 1/2
#TODO         x = pos + dx * widthx + dy * widthy
#TODO         n = normal + dx * widthx + dy * widthy
#TODO         g = metric(x)
#TODO         gu = inv(g)
#TODO         t = gu * Vec{T}(1, 0, 0, 0)
#TODO         t2 = t' * g * t
#TODO         n2 = n' * g * n
#TODO         u = (t / sqrt(-t2) + n / sqrt(n2)) / sqrt(T(2))
#TODO         Pixel{T}(x, u, zeros(SVector{3,T}))
#TODO     end
#TODO     canvas = Canvas{T}(Pixel{T}[make_pixel(i,j) for i in 1:ni, j in 1:nj])
#TODO 
#TODO     canvas = trace_rays(metric, objs, canvas)
#TODO 
#TODO     # scene = colorview(RGB,
#TODO     #                   T[mod(p.rgb[1] + 1/2, 1) for p in canvas.pixels]',
#TODO     #                   T[mod(p.rgb[2] + 1/2, 1) for p in canvas.pixels]',
#TODO     #                   T[mod(p.rgb[3] + 1/2, 1) for p in canvas.pixels]')
#TODO     scene = colorview(RGB,
#TODO                       T[p.rgb[1] for p in canvas.pixels]',
#TODO                       T[p.rgb[2] for p in canvas.pixels]',
#TODO                       T[p.rgb[3] for p in canvas.pixels]')
#TODO 
#TODO     mkpath(outdir)
#TODO     file = joinpath(outdir, "sphere2.png")
#TODO     rm(file, force=true)
#TODO     println("Output file is \"$file\"")
#TODO     save(file, scene)
#TODO     nothing
#TODO end

end
