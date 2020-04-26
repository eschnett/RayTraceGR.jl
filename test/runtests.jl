using RayTraceGR

using ComputedFieldTypes
using Grassmann
using LinearAlgebra
using StaticArrays
using Test



const Rat128 = Rational{Int128}
Base.rand(::Type{Rat128}) = Rat128(rand(Int16)) // 256
Base.rand(::Type{Rat128}, n::Integer) = Rat128[rand(Rat128) for i in 1:n]
Base.rand(::Type{Rat128}, n1::Integer, n2::Integer) =
    Rat128[rand(Rat128) for i in 1:n1, j in 1:n2]



@testset "Special values for D=$D" for D in 1:4
    V = SubManifold(Signature(D))
    T = Rat128

    Ch(T) = Chain{V,1,T}
    fullCh(T) = fulltype(Ch(T))

    @test zero(T)::T == 0
    rand(T)::T

    @test zero(Ch(T))::fullCh(T) == 0
    rand(Ch(T))::fullCh(T)

    @test zero(fullCh(T))::fullCh(T) == 0
    rand(fullCh(T))::fullCh(T)

    zero(Ch(Ch(T)))::fullCh(Ch(T)) # inefficient
    # @test zero(Ch(Ch(T)))::fullCh(Ch(T)) == 0
    rand(Ch(Ch(T)))::fullCh(Ch(T)) # inefficient

    @test zero(Ch(fullCh(T)))::fullCh(fullCh(T)) == 0
    rand(Ch(fullCh(T)))::fullCh(fullCh(T))

    zero(fullCh(Ch(T)))::fullCh(Ch(T)) # inefficient
    # @test zero(fullCh(Ch(T)))::fullCh(fullCh(T)) == 0
    rand(fullCh(Ch(T)))::fullCh(Ch(T)) # inefficient

    @test zero(fullCh(fullCh(T)))::fullCh(fullCh(T)) == 0
    rand(fullCh(fullCh(T)))::fullCh(fullCh(T))
end



@testset "Linear operators for D=$D" for D in 1:4
    V = SubManifold(Signature(D))
    T = Rat128
    Vec(X) = Chain{V,1,X}
    Mat(X) = Chain{V,1, fulltype(Vec(X))}

    for n in 1:100

        A = rand(Mat(T))
        x = rand(Vec(T))
        y = A ⋅ x
        y :: Vec(T)

        x1 = x.v
        A1 = SMatrix{D,D,T}((A.v[j].v[i] for i in 1:D, j in 1:D)...)
        y1 = A1 * x1
        y1 :: SVector{D,T}

        @test y.v == y1
    end

    # for n in 1:100
    #     A = rand(Mat(T))
    #     x = rand(Vec(T))
    #     y = rand(Vec(T))
    # 
    #     @test A ⋅ (x ∧ y) == (A ⋅ x) ∧ (A ⋅ y)
    # end
end



@testset "Minkowski metric" begin
    T = Rat128
    Vec(X) = Chain{V,1,X}
    Mat(X) = Chain{V,1, fulltype(Vec(X))}

    metric = minkowski

    x = Chain{V,1}(SVector{D,T}(0,0,0,0))
    g = metric(x)

    detg = detM(g)
    gu = invM(g)
    detgu = detM(gu)

    @test detg * detgu == 1

    for n in 1:100
        x = rand(Vec(T))
        @test gu ⋅ (g ⋅ x) == x
        @test g ⋅ (gu ⋅ x) == x
    end

    (g1,dg) = dmetric(metric, x)
    @test g1 == g
    @test all(==(0), dg)

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
