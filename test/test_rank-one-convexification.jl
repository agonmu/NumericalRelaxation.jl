
W_multi(F::Tensor{2,dim}) where dim = (norm(F)-1)^2
W_multi_rc(F::Tensor{2,dim}) where dim = norm(F) ≤ 1 ? 0.0 : (norm(F)-1)^2


##### multi-dimensional relaxation unit tests ####
@testset "Gradient Grid Iterator" begin
    # equidistant meshes
    gradientgrid_axes = -2.0:0.5:2
    gradientgrid1 = GradientGrid((gradientgrid_axes, gradientgrid_axes, gradientgrid_axes, gradientgrid_axes))
    @test size(gradientgrid1) == ntuple(x->length(gradientgrid_axes),4)
    @test size(gradientgrid1,1) == size(gradientgrid1,2) == size(gradientgrid1,3) == size(gradientgrid1,4)  == length(gradientgrid_axes)
    @test eltype(gradientgrid1) == Tensor{2,2,Float64,4}
    @test gradientgrid1[2,3,4,5] == Tensor{2,2}((gradientgrid_axes[2],gradientgrid_axes[3],gradientgrid_axes[4],gradientgrid_axes[5]))
    @test gradientgrid1[9,9,9,9] == gradientgrid1[end]
    @test gradientgrid1[1,1,1,1] == gradientgrid1[1]
    @test gradientgrid1[2] == gradientgrid1[2,1,1,1]
    @test gradientgrid1[9] == gradientgrid1[9,1,1,1]
    @test gradientgrid1[10] == gradientgrid1[1,2,1,1]
    @test NumericalRelaxation.center(gradientgrid1) == Tensor{2,2}((0.0,0.0,0.0,0.0))
    @test NumericalRelaxation.δ(gradientgrid1) == 0.5
    @test NumericalRelaxation.radius(gradientgrid1) == 2.0

    gradientgrid_axes = -1.0:0.5:2
    gradientgrid2 = GradientGrid((gradientgrid_axes, gradientgrid_axes, gradientgrid_axes, gradientgrid_axes))
    @test gradientgrid2[2,3,4,5] == Tensor{2,2}((gradientgrid_axes[2],gradientgrid_axes[3],gradientgrid_axes[4],gradientgrid_axes[5]))
    @test size(gradientgrid2) == ntuple(x->length(gradientgrid_axes),4)
    @test size(gradientgrid2,1) == size(gradientgrid2,2) == size(gradientgrid2,3) == size(gradientgrid2,4)  == length(gradientgrid_axes)
    @test NumericalRelaxation.center(gradientgrid2) == Tensor{2,2}((0.5,0.5,0.5,0.5))
    @test NumericalRelaxation.δ(gradientgrid2) == 0.5
    @test NumericalRelaxation.radius(gradientgrid2) == 1.5

    gradientgrid_axes = 1.0:0.1:2
    gradientgrid3 = GradientGrid((gradientgrid_axes, gradientgrid_axes, gradientgrid_axes, gradientgrid_axes))
    @test gradientgrid3[2,3,4,5] == Tensor{2,2}((gradientgrid_axes[2],gradientgrid_axes[3],gradientgrid_axes[4],gradientgrid_axes[5]))
    @test size(gradientgrid3) == ntuple(x->length(gradientgrid_axes),4)
    @test size(gradientgrid3,1) == size(gradientgrid3,2) == size(gradientgrid3,3) == size(gradientgrid3,4)  == length(gradientgrid_axes)
    @test NumericalRelaxation.center(gradientgrid3) == Tensor{2,2}((1.5,1.5,1.5,1.5))
    @test NumericalRelaxation.δ(gradientgrid3) == 0.1
    @test NumericalRelaxation.radius(gradientgrid3) == 0.5

    # non equidistant meshes
    gradientgrid_axes_diag = -2.0:0.5:2; gradientgrid_axes_off = -1.0:0.1:1.0
    gradientgrid1 = GradientGrid((gradientgrid_axes_diag, gradientgrid_axes_off, gradientgrid_axes_off, gradientgrid_axes_diag))
    @test eltype(gradientgrid1) == Tensor{2,2,Float64,4}
    @test size(gradientgrid1) == (length(gradientgrid_axes_diag),length(gradientgrid_axes_off),length(gradientgrid_axes_off),length(gradientgrid_axes_diag))
    @test size(gradientgrid1,1) == size(gradientgrid1,4) == length(gradientgrid_axes_diag)
    @test size(gradientgrid1,2) == size(gradientgrid1,3) == length(gradientgrid_axes_off)
    @test gradientgrid1[2,3,4,5] == Tensor{2,2}((gradientgrid_axes_diag[2],gradientgrid_axes_off[3],gradientgrid_axes_off[4],gradientgrid_axes_diag[5]))
    @test gradientgrid1[9,21,21,9] == gradientgrid1[end]
    @test gradientgrid1[1,1,1,1] == gradientgrid1[1]
    @test gradientgrid1[2] == gradientgrid1[2,1,1,1]
    @test gradientgrid1[9] == gradientgrid1[9,1,1,1]
    @test gradientgrid1[10] == gradientgrid1[1,2,1,1]
    @test NumericalRelaxation.δ(gradientgrid1) == 0.1
    @test NumericalRelaxation.center(gradientgrid1) == Tensor{2,2}((0.0,0.0,0.0,0.0))
    @test NumericalRelaxation.radius(gradientgrid1) == 2.0

    gradientgrid_axes_diag = -1.0:0.5:2; gradientgrid_axes_off = -3.0:0.1:0.0
    gradientgrid2 = GradientGrid((gradientgrid_axes_diag, gradientgrid_axes_off, gradientgrid_axes_off, gradientgrid_axes_diag))
    @test size(gradientgrid2) == (length(gradientgrid_axes_diag),length(gradientgrid_axes_off),length(gradientgrid_axes_off),length(gradientgrid_axes_diag))
    @test size(gradientgrid2,1) == size(gradientgrid2,4) == length(gradientgrid_axes_diag)
    @test size(gradientgrid2,2) == size(gradientgrid2,3) == length(gradientgrid_axes_off)
    @test NumericalRelaxation.center(gradientgrid2) == Tensor{2,2}((0.5,-1.5,-1.5,0.5))
    @test NumericalRelaxation.δ(gradientgrid2) == 0.1
    @test NumericalRelaxation.radius(gradientgrid2) == 1.5

    gradientgrid_axes_diag = 1.0:0.1:2; gradientgrid_axes_off = 0.0:0.1:2.0
    gradientgrid3 = GradientGrid((gradientgrid_axes_diag, gradientgrid_axes_off, gradientgrid_axes_off, gradientgrid_axes_diag))
    @test size(gradientgrid3) == (length(gradientgrid_axes_diag),length(gradientgrid_axes_off),length(gradientgrid_axes_off),length(gradientgrid_axes_diag))
    @test size(gradientgrid3,1) == size(gradientgrid3,4) == length(gradientgrid_axes_diag)
    @test size(gradientgrid3,2) == size(gradientgrid3,3) == length(gradientgrid_axes_off)
    @test NumericalRelaxation.center(gradientgrid3) == Tensor{2,2}((1.5,1.0,1.0,1.5))
    @test NumericalRelaxation.δ(gradientgrid3) == 0.1
    @test NumericalRelaxation.radius(gradientgrid3) == 1.0

    @test @inferred(gradientgrid1[1,1,1,1]) == Tensor{2,2}((-2.0,-1.0,-1.0,-2.0))
    @test @inferred(Union{Tensor{2,2},Nothing}, Base.iterate(gradientgrid1,1)) == (Tensor{2,2}((-2.0,-1.0,-1.0,-2.0)),2)
    @test @inferred(Union{Tensor{2,2},Nothing}, Base.iterate(gradientgrid1,100000)) === nothing
    @test @inferred(NumericalRelaxation.center(gradientgrid3)) == Tensor{2,2}((1.5,1.0,1.0,1.5))
    @test @inferred(NumericalRelaxation.δ(gradientgrid3)) == 0.1
    @test @inferred(NumericalRelaxation.radius(gradientgrid3)) == 1.0
    @test @inferred(size(gradientgrid3,1)) == size(gradientgrid3,4) == length(gradientgrid_axes_diag)
    @test size(gradientgrid3,1) == @inferred(size(gradientgrid3,4)) == length(gradientgrid_axes_diag)
    @test size(gradientgrid3,1) == size(gradientgrid3,4) == @inferred(length(gradientgrid_axes_diag))
end

@testset "Rank One Direction Iterator" begin
    gradientgrid_axes = -2.0:0.5:2
    gradientgrid1 = GradientGrid((gradientgrid_axes, gradientgrid_axes, gradientgrid_axes, gradientgrid_axes))
    dirs = ParametrizedR1Directions(2)
    @test(@inferred Union{Nothing,Tuple{Tuple{Vec{2,Int},Vec{2,Int}},Int}} Base.iterate(dirs,1) == ((Vec{2}((-1,-1)),Vec{2}((0,-1))),2))
    ((𝐚,𝐛),i) = Base.iterate(dirs,1)
    @test @inferred(NumericalRelaxation.inbounds_𝐚(gradientgrid1,𝐚)) && @inferred(NumericalRelaxation.inbounds_𝐛(gradientgrid1,𝐛))
end

@testset "R1Convexification" begin
    d = 2
    a = -2.0:0.5:2.0
    r1convexification_reduced = R1Convexification(a,a,dim=d,dirtype=ParametrizedR1Directions)
    buffer_reduced = build_buffer(r1convexification_reduced)
    convexify!(r1convexification_reduced,buffer_reduced,W_multi;buildtree=true)
    @test all(isapprox.(buffer_reduced.W_rk1.itp.itp.coefs .- [W_multi_rc(Tensor{2,2}((x1,x2,y1,y2))) for x1 in a, x2 in a, y1 in a, y2 in a],0.0,atol=1e-8))
    r1convexification_full = R1Convexification(a,a,dim=d,dirtype=ℛ¹Direction)
    buffer_full = build_buffer(r1convexification_full)
    convexify!(r1convexification_full,buffer_full,W_multi;buildtree=false)
    @test all(isapprox.(buffer_full.W_rk1.itp.itp.coefs .- [W_multi_rc(Tensor{2,2}((x1,x2,y1,y2))) for x1 in a, x2 in a, y1 in a, y2 in a],0.0,atol=1e-8))
    @test all(isapprox.(buffer_full.W_rk1.itp.itp.coefs .- buffer_reduced.W_rk1.itp.itp.coefs ,0.0,atol=1e-8))
    # test if subsequent convexifications work
    convexify!(r1convexification_full,buffer_full,W_multi;buildtree=true)
    @test all(isapprox.(buffer_full.W_rk1.itp.itp.coefs .- buffer_reduced.W_rk1.itp.itp.coefs ,0.0,atol=1e-8))

    @testset "Tree Construction" begin
        F1 = Tensor{2,2}((0.1,0.0,0.0,0.0))
        F2 = Tensor{2,2}((0.1,0.5,0.3,0.2))
        F3 = Tensor{2,2}((0.5,0.5,0.0,0.0))
        for F in (F1,F2,F3)
            flt = FlexibleLaminateTree(F,r1convexification_full,buffer_full,3)
            @test NumericalRelaxation.checkintegrity(flt,buffer_full.W_rk1)
            𝔸, 𝐏, W = NumericalRelaxation.eval(flt,W_multi)
            @test W == 0.0
            @test 𝐏 == zero(Tensor{2,2})
        end
    end
end