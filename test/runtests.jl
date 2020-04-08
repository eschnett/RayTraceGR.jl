using RayTraceGR

using LinearAlgebra
using StaticArrays
using Test

const BigRat = Rational{BigInt}

@testset "metric" for T in [BigRat]
    metric = minkowski

    x = Vec{T}(0, 0, 0, 0)
    g = metric(x)

    detg = det(g)
    gu = inv(g)
    detgu = det(gu)

    @test detg * detgu == 1
    @test g * gu == I

    Γ = christoffel(metric, x)
    @test all(==(0), Γ)
end



@testset "rays" for T in [Float32]
    metric = minkowski
    x = Vec{T}(0, 0, 0, 0)
    n = Vec{T}(0, 1, 0, 0)
    p = Pixel{T}(x, n, zeros(SVector{3,T}))
    objs = Object[]
    p = trace_ray(metric, objs, p)
    @test maximum(abs.(p.rgb - [10, 0, 0])) <= 1.0e-5
end

