using RayTraceGR

using DifferentialEquations
using LinearAlgebra
using StaticArrays
using Test

const BigRat = Rational{BigInt}

@testset "metric" for T in [BigRat]
    x = Vec{T}(0, 0, 0, 0)
    g = metric(x)

    detg = det(g)
    gu = inv(g)
    detgu = det(gu)

    @test detg * detgu == 1
    @test g * gu == I

    Γ = christoffel(x)
end



@testset "rays" for T in [Float32]
    x = Vec{T}(0, 0, 0, 0)
    n = Vec{T}(0, 1, 0, 0)
    p = Pixel{T}(x, n, zeros(SVector{3,T}))
    p = trace_ray(p)
    @test maximum(abs.(p.rgb - [1, 0, 0])) <= 1.0e-6
end

