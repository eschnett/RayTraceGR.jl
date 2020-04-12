using RayTraceGR

using LinearAlgebra
using StaticArrays
using Test



const BigRat = Rational{BigInt}

@testset "Minkowski metric" begin
    T = BigRat
    metric = minkowski

    x = Vec{T}(0, 0, 0, 0)
    g = metric(x)

    detg = det(g)
    gu = inv(g)
    detgu = det(gu)

    @test detg * detgu == 1
    @test g * gu == I

    (g1,dg) = dmetric(metric, x)
    @test g1 == g
    @test all(==(0), dg)

    Γ = christoffel(metric, x)
    @test all(==(0), Γ)
end



@testset "Kerr-Schild metric" for i in 1:7
    T = Float32
    tol = eps(T)^(T(3)/4)
    metric = kerr_schild

    ix = i & 1
    iy = i & 2
    iz = i & 4
    x = Vec{T}(0, 2ix, 2iy, 2iz)

    g = metric(x)
    @test !any(isnan, g)

    detg = det(g)
    gu = inv(g)
    detgu = det(gu)

    @test abs(detg * detgu - 1) <= tol
    @test maximum(abs.(g * gu - I)) <= tol

    (g1,dg) = dmetric(metric, x)
    @test maximum(abs.(g - metric(x))) <= tol

    Γ = christoffel(metric, x)
    @test !any(isnan, Γ)
end



@testset "rays" begin
    T = Float32
    tol = eps(T)^(T(3)/4)
    metric = minkowski
    x = Vec{T}(0, 0, 0, 0)
    u = Vec{T}(-1, 1, 0, 0)
    p = Pixel{T}(x, u, zeros(SVector{3,T}))
    objs = Object{T}[]
    p = trace_ray(metric, objs, p)
    @test maximum(abs.(p.rgb - [10, 0, 0])) <= tol
end
