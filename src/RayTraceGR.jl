module RayTraceGR

# using Distributed
using Images
using LinearAlgebra
using OrdinaryDiffEq
# using SharedArrays
using StaticArrays



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

export S; const S = S"-+++"     # signature

@mixedbasis S
# defines V
# defines (⟨-++++---⟩*, v, v₁, v₂, v₃, v₄, w¹, w², w³, w⁴, v₁₂, v₁₃,
# v₁₄, v₁w¹, v₁w², v₁w³, v₁w⁴, v₂₃, v₂₄, v₂w¹, v₂w², v₂w³, v₂w⁴, v₃₄,
# v₃w¹, v₃w², v₃w³, v₃w⁴, v₄w¹, v₄w², v₄w³, v₄w⁴, w¹², w¹³, w¹⁴, w²³,
# w²⁴, w³⁴, v₁₂₃, v₁₂₄, v₁₂w¹, v₁₂w², v₁₂w³, v₁₂w⁴, v₁₃₄, v₁₃w¹,
# v₁₃w², v₁₃w³, v₁₃w⁴, v₁₄w¹, v₁₄w², v₁₄w³, v₁₄w⁴, v₁w¹², v₁w¹³,
# v₁w¹⁴, v₁w²³, v₁w²⁴, v₁w³⁴, v₂₃₄, v₂₃w¹, v₂₃w², v₂₃w³, v₂₃w⁴, v₂₄w¹,
# v₂₄w², v₂₄w³, v₂₄w⁴, v₂w¹², v₂w¹³, v₂w¹⁴, v₂w²³, v₂w²⁴, v₂w³⁴,
# v₃₄w¹, v₃₄w², v₃₄w³, v₃₄w⁴, v₃w¹², v₃w¹³, v₃w¹⁴, v₃w²³, v₃w²⁴,
# v₃w³⁴, v₄w¹², v₄w¹³, v₄w¹⁴, v₄w²³, v₄w²⁴, v₄w³⁴, w¹²³, w¹²⁴, w¹³⁴,
# w²³⁴, v₁₂₃₄, v₁₂₃w¹, v₁₂₃w², v₁₂₃w³, v₁₂₃w⁴, v₁₂₄w¹, v₁₂₄w², v₁₂₄w³,
# v₁₂₄w⁴, v₁₂w¹², v₁₂w¹³, v₁₂w¹⁴, v₁₂w²³, v₁₂w²⁴, v₁₂w³⁴, v₁₃₄w¹,
# v₁₃₄w², v₁₃₄w³, v₁₃₄w⁴, v₁₃w¹², v₁₃w¹³, v₁₃w¹⁴, v₁₃w²³, v₁₃w²⁴,
# v₁₃w³⁴, v₁₄w¹², v₁₄w¹³, v₁₄w¹⁴, v₁₄w²³, v₁₄w²⁴, v₁₄w³⁴, v₁w¹²³,
# v₁w¹²⁴, v₁w¹³⁴, v₁w²³⁴, v₂₃₄w¹, v₂₃₄w², v₂₃₄w³, v₂₃₄w⁴, v₂₃w¹²,
# v₂₃w¹³, v₂₃w¹⁴, v₂₃w²³, v₂₃w²⁴, v₂₃w³⁴, v₂₄w¹², v₂₄w¹³, v₂₄w¹⁴,
# v₂₄w²³, v₂₄w²⁴, v₂₄w³⁴, v₂w¹²³, v₂w¹²⁴, v₂w¹³⁴, v₂w²³⁴, v₃₄w¹²,
# v₃₄w¹³, v₃₄w¹⁴, v₃₄w²³, v₃₄w²⁴, v₃₄w³⁴, v₃w¹²³, v₃w¹²⁴, v₃w¹³⁴,
# v₃w²³⁴, v₄w¹²³, v₄w¹²⁴, v₄w¹³⁴, v₄w²³⁴, w¹²³⁴, v₁₂₃₄w¹, v₁₂₃₄w²,
# v₁₂₃₄w³, v₁₂₃₄w⁴, v₁₂₃w¹², v₁₂₃w¹³, v₁₂₃w¹⁴, v₁₂₃w²³, v₁₂₃w²⁴,
# v₁₂₃w³⁴, v₁₂₄w¹², v₁₂₄w¹³, v₁₂₄w¹⁴, v₁₂₄w²³, v₁₂₄w²⁴, v₁₂₄w³⁴,
# v₁₂w¹²³, v₁₂w¹²⁴, v₁₂w¹³⁴, v₁₂w²³⁴, v₁₃₄w¹², v₁₃₄w¹³, v₁₃₄w¹⁴,
# v₁₃₄w²³, v₁₃₄w²⁴, v₁₃₄w³⁴, v₁₃w¹²³, v₁₃w¹²⁴, v₁₃w¹³⁴, v₁₃w²³⁴,
# v₁₄w¹²³, v₁₄w¹²⁴, v₁₄w¹³⁴, v₁₄w²³⁴, v₁w¹²³⁴, v₂₃₄w¹², v₂₃₄w¹³,
# v₂₃₄w¹⁴, v₂₃₄w²³, v₂₃₄w²⁴, v₂₃₄w³⁴, v₂₃w¹²³, v₂₃w¹²⁴, v₂₃w¹³⁴,
# v₂₃w²³⁴, v₂₄w¹²³, v₂₄w¹²⁴, v₂₄w¹³⁴, v₂₄w²³⁴, v₂w¹²³⁴, v₃₄w¹²³,
# v₃₄w¹²⁴, v₃₄w¹³⁴, v₃₄w²³⁴, v₃w¹²³⁴, v₄w¹²³⁴, v₁₂₃₄w¹², v₁₂₃₄w¹³,
# v₁₂₃₄w¹⁴, v₁₂₃₄w²³, v₁₂₃₄w²⁴, v₁₂₃₄w³⁴, v₁₂₃w¹²³, v₁₂₃w¹²⁴,
# v₁₂₃w¹³⁴, v₁₂₃w²³⁴, v₁₂₄w¹²³, v₁₂₄w¹²⁴, v₁₂₄w¹³⁴, v₁₂₄w²³⁴,
# v₁₂w¹²³⁴, v₁₃₄w¹²³, v₁₃₄w¹²⁴, v₁₃₄w¹³⁴, v₁₃₄w²³⁴, v₁₃w¹²³⁴,
# v₁₄w¹²³⁴, v₂₃₄w¹²³, v₂₃₄w¹²⁴, v₂₃₄w¹³⁴, v₂₃₄w²³⁴, v₂₃w¹²³⁴,
# v₂₄w¹²³⁴, v₃₄w¹²³⁴, v₁₂₃₄w¹²³, v₁₂₃₄w¹²⁴, v₁₂₃₄w¹³⁴, v₁₂₃₄w²³⁴,
# v₁₂₃w¹²³⁴, v₁₂₄w¹²³⁴, v₁₃₄w¹²³⁴, v₂₃₄w¹²³⁴, v₁₂₃₄w¹²³⁴)



# export V; const V = Λ(S)                    # basis
# export M; const M = typeof(V).parameters[1] # manifold

# @basis @
# # defines ⟨-+++⟩, v, v₁, v₂, v₃, v₄, v₁₂, v₁₃, v₁₄, v₂₃, v₂₄, v₃₄,
# # v₁₂₃, v₁₂₄, v₁₃₄, v₂₃₄, v₁₂₃₄


# const Vec = TensorGraded{M, 1}
# const Mat = TensorGraded{M⊕M', 2}

const Vec{T} = Chain{M, 1, T} where {T}
const Mat{T} = Chain{M⊕M', 2, T} where {T}



# Λ(S)
# Λ(S)'
# 
# Λ(S).v1
# Λ(S)'.w2
# Λ(S).v1 ∧ Λ(S)'.w2
# Λ(S)'.w2(Λ(S).v1)



################################################################################



export minkowski
"""
Minkowski metric g_ab
"""
function minkowski(x::Vec{T})::Mat{T} where {T}
    # smat(T, (a,b) -> a==b ? (a==1 ? -1 : 1) : 0)
    -v1w1 + v2w2 + v3w3 + v4w4
end



################################################################################

# COMPATIBILITY
export D
const D = ndims(M)



# OUTDATED

export Vec, Mat, Ten3
const Vec{T} = SVector{D,T}
const Mat{T} = SMatrix{D,D,T}
const Ten3{T} = SArray{Tuple{D,D,D}, T}



export svec, smat, sten3
function svec(::Type{T}, f::F) where {T, F}
    @SArray T[f(a) for a in 1:D]
end
function smat(::Type{T}, f::F) where {T, F}
    @SArray T[f(a, b) for a in 1:D, b in 1:D]
end
function sten3(::Type{T}, f::F) where {T, F}
    @SArray T[f(a, b, c) for a in 1:D, b in 1:D, c in 1:D]
end

export ssum, ssum2, ssum3
function ssum(::Type{T}, f::F, r::R=1:D) where {T, F, R}
    s = T(0)
    for i in r
        s += f(i)::T
    end
    s
end
function ssum2(::Type{T}, f::F, r::R=1:D) where {T, F, R}
    s = T(0)
    for i in r, j in r
        s += f(i, j)::T
    end
    s
end
function ssum3(::Type{T}, f::F, r::R=1:D) where {T, F, R}
    s = T(0)
    for i in r, j in r, k in r
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
    smat(T, (a,b) -> a==b ? (a==1 ? -1 : 1) : 0)
end



export kerr_schild
"""
Kerr-Schild metric g_ab

Living Reviews in Relativity, Greg Cook, 2000, section 3.3.1
"""
function kerr_schild(xx::Vec{T})::Mat{T} where {T}
    M = 1
    a = 0                       # T(0.8)

    t,x,y,z = xx
    @assert !any(isnan, (t, x, y, z))

    # <https://en.wikipedia.org/wiki/Kerr_metric>
    η = smat(T, (a,b) -> a==b ? (a==1 ? -1 : 1) : 0)
    ρ = sqrt(x^2 + y^2 + z^2)
    r = sqrt(ρ^2 - a^2)/2 + sqrt(a^2*z^2 + ((ρ^2 - a^2)/2)^2)
    f = 2*M*r^3 / (r^4 + a^2*z^2)
    k = @SVector T[1,
                   (r*x + a*y) / (r^2 + a^2),
                   (r*y - a*x) / (r^2 + a^2),
                   z / r]

    g = smat(T, (a,b) -> η[a,b] + f * k[a] * k[b])

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
    g = smat(T, (a,b) -> gdg[a,b].val)
    dg = sten3(T, (a,b,c) -> gdg[a,b].eps[c])
    (g, dg)
end



export christoffel
"""
Christoffel symbol Γ^a_bc
"""
function christoffel(metric::Metric, x::Vec{T})::Ten3{T} where {Metric, T}
    g,dg = dmetric(metric, x)
    gu = inv(g)
    Γl = sten3(T, (a,b,c) -> (dg[a,b,c] + dg[a,c,b] - dg[b,c,a]) / 2)
    sten3(T, (a,b,c) -> ssum(T, x -> gu[a,x] * Γl[x,b,c]))
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
    udot = svec(T, a -> - ssum2(T, (x,y) -> Γ[a,x,y] * r.u[x] * r.u[y]))
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
function objcolor(obj::Object, pos::Vec)
    @error "Called color on abstract object"
end



export Plane
struct Plane{T} <: Object{T}
    # TOOD: add normal
    time::T
end

function distance(pl::Plane{T}, pos::Vec{T})::T where{T}
    pos[1] - pl.time
end
function objcolor(pl::Plane{T}, pos::Vec{T})::SVector{3,T} where{T}
    SVector{3,T}(0, T(1)/2, 0)
end



export Sphere
struct Sphere{T} <: Object{T}
    pos::Vec{T}
    vel::Vec{T}
    radius::T
end

function distance(sph::Sphere{T}, pos::Vec{T})::T where{T}
    # TODO: Use metric?
    sign(sph.radius) *
        (ssum(T, a -> (pos[a] - sph.pos[a])^2, 2:D) - sph.radius^2)
end
function objcolor(sph::Sphere{T}, pos::Vec{T})::SVector{3,T} where{T}
    x = pos[2] - sph.pos[2]
    y = pos[3] - sph.pos[3]
    z = pos[4] - sph.pos[4]
    r = sqrt(x^2 + y^2 + z^2)
    θ = acos(z / r)
    ϕ = atan(y, x)
    SVector{3,T}(mod(12 * θ / π, 1), mod(12 * ϕ / π, 1), T(1))
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
    @assert abs(u2) <= tol

    λ0 = T(0)
    λ1 = T(100)
    r = Ray{T}(x, u)
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
    sol = solve(prob, Tsit5(), callback=cb, reltol=tol, abstol=tol)
    λend = sol.t[end]
    rend = s2r(sol(λend))
    x,u = rend.x,rend.u

    # g = metric(x)
    # u2 = u' * g * u
    # rgb = @view rend.x[2:4]
    # Pixel{T}(p.pos, p.normal, rgb)

    imin = 0
    dmin = T(0.01)
    for (i,obj) in enumerate(objs)
        d = distance(obj, x)
        if d < dmin
            imin = i
            dmin = d
        end
    end
    if imin == 0
        col = SVector{3,T}(1, 0, 0) # did not hit any object
    else
        col = objcolor(objs[imin], x) .* (T(imin) / length(objs))
    end
    Pixel{T}(p.pos, p.normal, col)
end



export Canvas
struct Canvas{T}
    pixels::Array{Pixel{T}, 2}
end

export trace_rays
function trace_rays(metric, objs::Vector{Object{T}},
                    c::Canvas{T})::Canvas{T} where {T}
    l = length(c.pixels)

    # Serial
    # c = Canvas{T}(Pixel{T}[
    #     (
    # print("\r$(round(Int, 100*(i-1)/l))%");
    # trace_ray(metric, objs, p)
    # )
    #     for (i,p) in enumerate(c.pixels)])
    # println("\r100%")

    # Distributed
    # ps = SharedArray{Pixel{T}}(l)
    # @sync @distributed for i in 1:l
    #     ps[i] = trace_ray(metric, objs, c.pixels[i])
    # end
    # c = Canvas{T}(Array{Pixel{T}, 2}(undef, size(c.pixels)))
    # for i in 1:l
    #     c.pixels[i] = ps[i]
    # end

    # Multi-threaded per pixel
    # ps = Array{Pixel{T}, 2}(undef, size(c.pixels))
    # Threads.@threads for i in 1:l
    #     ps[i] = trace_ray(metric, objs, c.pixels[i])
    # end
    # c = Canvas{T}(ps)

    # Multi-threaded per scanline
    ni,nj = size(c.pixels)
    ps = Array{Pixel{T}, 2}(undef, ni, nj)
    Threads.@threads for j in 1:nj
        for i in 1:ni
            ps[i,j] = trace_ray(metric, objs, c.pixels[i,j])
        end
    end
    c = Canvas{T}(ps)

    c
end



const outdir = "scenes"

function example1()
    T = Float64

    metric = minkowski
    caelum = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(1, 0, 0, 0), -10)
    frustum = Plane{T}(-20)
    sphere = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(1, 0, 0, 0), T(1)/2)
    objs = Object{T}[caelum, frustum, sphere]

    ni = 200
    nj = 200
    pos = Vec{T}(0, 0, -2, 0)
    widthx = Vec{T}(0, 1, 0, 0)
    widthy = Vec{T}(0, 0, 0, 1)
    normal = Vec{T}(0, 0, 1, 0)
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
    nothing
end

function example2()
    T = Float64

    metric = kerr_schild
    caelum = Sphere{T}(Vec{T}(0, 0, 0, 0), Vec{T}(1, 0, 0, 0), -10)
    frustum = Plane{T}(-20)
    sphere = Sphere{T}(Vec{T}(0, 4, 0, 0), Vec{T}(1, 0, 0, 0), T(1)/2)
    objs = Object{T}[caelum, frustum, sphere]

    ni = 200
    nj = 200
    # TODO: make screen infinitesimally small, pixels only differ by
    # their normal direction
    pos = Vec{T}(0, 4, -2, 0)
    widthx = Vec{T}(0, 1, 0, 0)
    widthy = Vec{T}(0, 0, 0, 1)
    normal = Vec{T}(0, 0, 1, 0)
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
    nothing
end

end
