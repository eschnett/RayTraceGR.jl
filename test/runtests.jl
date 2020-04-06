using RayTraceGR

using DifferentialEquations
using LinearAlgebra
using StaticArrays

const BigRat = Rational{BigInt}

# const T = BigRat
const T = Float64

x = Vec{T}([0, 0, 0, 0])
@show x
g = metric(x)
@show g

detg = det(g)
@show detg
gu = inv(g)
@show gu

Γ = christoffel(x)
@show Γ

λ = T(0)
v = Vec{T}([-1, 1, 0, 0])
r = Ray{T}(x, v)
@show r
rdot = geodesic(r, nothing, λ)
@show rdot

tspan = (T(0), T(1))
prob = ODEProblem(geodesic, r2s(r), tspan)
@show "prob"
@show prob
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
@show "sol"
@show sol
@show sol(0)
@show sol(1)
@show sol(0.7)
