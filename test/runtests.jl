using RayTraceGR

using Grassmann
using LinearAlgebra
# using StaticArrays
using Test



const Rat128 = Rational{Int128}
Base.rand(::Type{Rat128}) = rand(Int16) // 256
Base.rand(::Type{Rat128}, n::Integer) = Rat128[rand(Rat128) for i in 1:n]
Base.rand(::Type{Rat128}, n1::Integer, n2::Integer) =
    Rat128[rand(Rat128) for i in 1:n1, j in 1:n2]



@testset "Linear operators" begin
    T = Rat128
    for n in 1:100
        A1 = rand(T, D, D)
        A1::Matrix{T}
        x1 = rand(T, D)
        x1::Vector{T}
        y1 = A1 * x1
        y1::Vector{T}
        A = makeMat(T, j->sum(Vec{T}, i->makeVec(T, A1[i,j]*v(i)), 1:D))
        A::Mat{T}
        x = sum(Vec{T}, i->makeVec(T, x1[i]*v(i)), 1:D)
        x::Vec{T}
        y = A(x)
        y::Vec{T}
        y0 = sum(Vec{T}, i->makeVec(T, y1[i]*v(i)), 1:D)
        y0::Vec{T}
        @test y == y0
    end
end



@testset "Minkowski metric" begin
    T = Rat128
    metric = minkowski

    x = makeVec(T, d->0)
    g = metric(x)

    detg = det(g)
    gu = inv(g)
    detgu = det(gu)

    @test detg * detgu == 1
    for n in 1:100
        x = rand(Vec{T})
        @test gu(g(x)) == x
        @test g(gu(x)) == x
    end

    # (g1,dg) = dmetric(metric, x)
    # @test g1 == g
    # @test all(==(0), dg)
    # 
    # Γ = christoffel(metric, x)
    # @test all(==(0), Γ)
end



# @testset "Kerr-Schild metric" for i in 1:7
#     T = Float32
#     tol = eps(T)^(T(3)/4)
#     metric = kerr_schild
# 
#     ix = i & 1
#     iy = i & 2
#     iz = i & 4
#     x = Vec{T}(0, 2ix, 2iy, 2iz)
# 
#     g = metric(x)
#     @test !any(isnan, g)
# 
#     detg = det(g)
#     gu = inv(g)
#     detgu = det(gu)
# 
#     @test abs(detg * detgu - 1) <= tol
#     @test maximum(abs.(g * gu - I)) <= tol
# 
#     (g1,dg) = dmetric(metric, x)
#     @test maximum(abs.(g - metric(x))) <= tol
# 
#     Γ = christoffel(metric, x)
#     @test !any(isnan, Γ)
# end
# 
# 
# 
# @testset "rays" begin
#     T = Float32
#     tol = eps(T)^(T(3)/4)
#     metric = minkowski
#     x = Vec{T}(0, 0, 0, 0)
#     u = Vec{T}(-1, 1, 0, 0)
#     p = Pixel{T}(x, u, zeros(SVector{3,T}))
#     objs = Object{T}[]
#     p = trace_ray(metric, objs, p)
#     # @test maximum(abs.(p.rgb - [10, 0, 0])) <= tol
#     @test maximum(abs.(p.rgb - [1, 0, 0])) <= tol
# end
