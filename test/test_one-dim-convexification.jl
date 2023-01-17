W(F::Number,x1=2,x2=6) = (F-1)^x1 * (42 + 77*F + 15*F^x1 - 102.5*F^3 + 58.89*F^4 - 12.89*F^5 + F^x2)
W(F::Tensor{2,1},x1=2,x2=6) = W(F[1],x1,x2)

@testset "Equidistant Graham Scan" begin
    convexification = GrahamScan(start=0.01,stop=5.0,δ=0.01)
    buffer = build_buffer(convexification) 
    W_conv, F⁺, F⁻ = convexify(convexification,buffer,W,Tensor{2,1}((2.0,)))
    @test isapprox(W_conv,7.2,atol=1e-1)
    @test isapprox(F⁺[1],4.0,atol=1e-1)
    @test isapprox(F⁻[1],1.0,atol=1e-1)
    @test @inferred(convexify(convexification,buffer,W,Tensor{2,1}((2.0,)))) == (W_conv,F⁺,F⁻)
end

@testset "QHull" begin
    convexification = QHull(start=0.01,stop=5.0,δ=0.01)
    buffer = build_buffer(convexification) 
    W_conv, F⁺, F⁻ = convexify(convexification,buffer,W,Tensor{2,1}((2.0,)))
    @test isapprox(W_conv,7.2,atol=1e-1)
    @test isapprox(F⁺[1],4.0,atol=1e-1)
    @test isapprox(F⁻[1],1.0,atol=1e-1)
    @test @inferred(convexify(convexification,buffer,W,Tensor{2,1}((2.0,)))) == (W_conv,F⁺,F⁻)
end

@testset "Adaptive GrahamScan Convexification" begin
    ac = AdaptiveGrahamScan(
            interval=[0.001,5.0],
            basegrid_numpoints=50,
            adaptivegrid_numpoints=115,
            exponent=5,
            distribution="fix",
            stepSizeIgnoreHessian=0.05,
            minPointsPerInterval=15,
            radius=3,
            minStepSize=0.03,
            forceAdaptivity=false)
    @testset "build_buffer()" begin
        buf = @inferred NumericalRelaxation.build_buffer(ac)
        @test typeof(buf) == (NumericalRelaxation.AdaptiveConvexificationBuffer1D{Tensor{2,1,Float64,1},Float64,Tensor{4,1,Float64,1}})
        @test length(buf.basebuffer.grid) == ac.basegrid_numpoints
        @test length(buf.basebuffer.values) == ac.basegrid_numpoints
        @test length(buf.basegrid_∂²W) == ac.basegrid_numpoints
        @test length(buf.adaptivebuffer.grid) == ac.adaptivegrid_numpoints
        @test length(buf.adaptivebuffer.values) == ac.adaptivegrid_numpoints
        @test buf.basebuffer.grid[1][1] == ac.interval[1]
        @test buf.basebuffer.grid[end][1] == ac.interval[2]
        δ_sum = 0
        for i in 1:ac.basegrid_numpoints-1 #check for equal distribution
            δ_sum = abs((buf.basebuffer.grid[i+1][1]-buf.basebuffer.grid[i][1])-(ac.interval[2]-ac.interval[1])/(ac.basegrid_numpoints-1))
        end
        @test isapprox(δ_sum,0,atol=1e-10)
    end
    @testset "convexify!()" begin
        buffer = build_buffer(ac)
        F = Tensors.Tensor{2,1}((2.0,))
        W_conv, F⁺, F⁻ = convexify(ac,buffer,W,Tensor{2,1}((2.0,)))
        @test isapprox(W_conv,7.2,atol=1e-1)
        @test isapprox(F⁺[1],4.0,atol=1e-1)
        @test isapprox(F⁻[1],1.0,atol=1e-1)
        @test @inferred(convexify(ac,buffer,W,Tensor{2,1}((2.0,)))) == (W_conv,F⁺,F⁻)
    end
end

@testset "Adaptive QHull & GrahamScan Convexification" begin
    ac = AdaptiveQHull(
            interval=[0.001,5.0],
            basegrid_numpoints=50,
            adaptivegrid_numpoints=115,
            exponent=5,
            distribution="fix",
            stepSizeIgnoreHessian=0.05,
            minPointsPerInterval=15,
            radius=3,
            minStepSize=0.03,
            forceAdaptivity=false)
    @testset "build_buffer()" begin
        buf = @inferred NumericalRelaxation.build_buffer(ac)
        @test typeof(buf) == (NumericalRelaxation.AdaptiveConvexificationBuffer1D{Tensor{2,1,Float64,1},Float64,Tensor{4,1,Float64,1}})
        @test length(buf.basebuffer.grid) == ac.basegrid_numpoints
        @test length(buf.basebuffer.values) == ac.basegrid_numpoints
        @test length(buf.basegrid_∂²W) == ac.basegrid_numpoints
        @test length(buf.adaptivebuffer.grid) == ac.adaptivegrid_numpoints
        @test length(buf.adaptivebuffer.values) == ac.adaptivegrid_numpoints
        @test buf.basebuffer.grid[1][1] == ac.interval[1]
        @test buf.basebuffer.grid[end][1] == ac.interval[2]
        δ_sum = 0
        for i in 1:ac.basegrid_numpoints-1 #check for equal distribution
            δ_sum = abs((buf.basebuffer.grid[i+1][1]-buf.basebuffer.grid[i][1])-(ac.interval[2]-ac.interval[1])/(ac.basegrid_numpoints-1))
        end
        @test isapprox(δ_sum,0,atol=1e-10)
    end
    @testset "convexify!()" begin
        buffer = build_buffer(ac)
        F = Tensors.Tensor{2,1}((2.0,))
        W_conv, F⁺, F⁻ = convexify(ac,buffer,W,Tensor{2,1}((2.0,)))
        @test isapprox(W_conv,7.2,atol=1e-1)
        @test isapprox(F⁺[1],4.0,atol=1e-1)
        @test isapprox(F⁻[1],1.0,atol=1e-1)
        @test @inferred(convexify(ac,buffer,W,Tensor{2,1}((2.0,)))) == (W_conv,F⁺,F⁻)
    end
end
#    @testset "adaptive_1Dgrid!()" begin
#        for material in materialvec
#            buf = ConvexDamage.build_buffer(material.convexstrategy)
#            state = ConvexDamage.init_materialstate(material)
#            state.convexificationbuffer.basebuffer.values .= [ConvexDamage.W_energy(x, material, state.damage) for x in state.convexificationbuffer.basebuffer.grid]
#            state.convexificationbuffer.basegrid_∂²W .= [Tensors.hessian(i->ConvexDamage.W_energy(i,material,state.damage), x) for x in state.convexificationbuffer.basebuffer.grid]
#            Fpm = @inferred ConvexDamage.adaptive_1Dgrid!(material.convexstrategy,state.convexificationbuffer)
#            gridnorm = norm(getindex.(state.convexificationbuffer.adaptivebuffer.grid,1))
#            @test isapprox(gridnorm,119.80529;atol=1e-5)
#            for (i,f) in enumerate([0.001,0.24589,0.7356,1.2254,14.6948,20.001])
#                @test isapprox(Fpm[i],Tensors.Tensor{2,1}((f,));atol=1e-4)
#            end
#            #same result for subsequent calls?
#            @test @inferred(ConvexDamage.adaptive_1Dgrid!(material,state)) == Fpm
#            @test norm(getindex.(state.convexificationbuffer.adaptivebuffer.grid,1)) == gridnorm
#        end
#    end
#    @testset "check_hessian()"
#    begin
#        for material in materialvec
#            buf = ConvexDamage.build_buffer(material.convexstrategy)
#            state = ConvexDamage.init_materialstate(material)
#            state.convexificationbuffer.basegrid_∂²W .= [Tensors.hessian(i->ConvexDamage.W_energy(i,material,state.damage), x) for x in state.convexificationbuffer.basebuffer.grid]
#            @inferred ConvexDamage.check_hessian(material.convexstrategy,state.convexificationbuffer)
#            F_hes = ConvexDamage.check_hessian(state.convexificationbuffer.basegrid_∂²W, state.convexificationbuffer.basebuffer.grid, material.convexstrategy)
#            for (i,F) in enumerate([0.40916 2.44997])
#                @test isapprox(F_hes[i],Tensors.Tensor{2,1}((F,));atol=1e-4)
#            end
#            @test ConvexDamage.check_hessian(state.convexificationbuffer.basegrid_∂²W, state.convexificationbuffer.basebuffer.grid, material.convexstrategy) == F_hes
#        end
#        # check for different data types
#        for T_F in [Float64, Tensors.Tensor{2,1,Float64,1}]
#            for T_∂²W in [Float64, Tensors.Tensor{4,1,Float64,1}]
#                F = ones(T_F,11).*      [0.0, 1.0, 1.05, 1.1, 1.14999, 1.19998, 6.0, 7.0, 8.0, 9.0, 10.0]
#                ∂²W = ones(T_∂²W,11).*  [1.0, 2.0, 1.00, 2.0, 1.00000, 2.00000, 2.0, 1.9, 2.0, 1.9, 1.9]
#                @test ConvexDamage.check_hessian(∂²W, F,ac) == ones(T_F,2).*[1.05, 7.0]
#            end
#        end
#    end
#    @testset "check_slope()" begin
#        for material in materialvec
#            buf = ConvexDamage.build_buffer(material.convexstrategy)
#            state = ConvexDamage.init_materialstate(material)
#            state.convexificationbuffer.basebuffer.values .= [ConvexDamage.W_energy(x, material, state.damage) for x in state.convexificationbuffer.basebuffer.grid]
#            @inferred ConvexDamage.check_slope(state.convexificationbuffer)
#            F_slp = @inferred ConvexDamage.check_slope(state.convexificationbuffer.basebuffer.grid, state.convexificationbuffer.basebuffer.values)
#            for (i,F) in enumerate([0.001, 1.2254, 14.6948, 20.001])
#                @test isapprox(F_slp[i],Tensors.Tensor{2,1}((F,));atol=1e-4)
#            end
#            # subsequent call returns same result??
#            @test ConvexDamage.check_slope(state.convexificationbuffer.basebuffer.grid, state.convexificationbuffer.basebuffer.values) == F_slp
#        end
#        # test for different datatypes
#        for T_F in [Float64, Tensors.Tensor{2,1,Float64,1}]
#            for T_W in [Float64, Tensors.Tensor{4,1,Float64,1}]
#                F = ones(T_F,11).*  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
#                W = ones(T_W,11).*  [9.0, 8.0, 7.0, 6.1, 5.3, 4.6, 5.0, 6.0, 7.1, 8.0, 9.0]
#                @test ConvexDamage.check_slope(F, W) == ones(T_F,4).*[0.0, 7.0, 9.0, 10.0]
#            end
#        end
#        # ====== check helper fcns ==========
#        # iterator()
#        mask = ones(Bool,10); mask[2:6].=zeros(Bool,5);
#        @inferred ConvexDamage.iterator(4,mask;dir=1)
#        @test isapprox(ConvexDamage.iterator(1,mask;dir=1),7;atol=0)
#        @test isapprox(ConvexDamage.iterator(7,mask;dir=-1),1;atol=0)
#        @test isapprox(ConvexDamage.iterator(1,mask;dir=-1),1;atol=0)
#        @test isapprox(ConvexDamage.iterator(10,mask;dir=1),10;atol=0)
#        # is_convex()
#        for T1 in [Tensors.Tensor{2,1}, Tensors.Tensor{4,1}, Float64]
#            for T2 in [Tensors.Tensor{2,1}, Tensors.Tensor{4,1}, Float64]
#                P1 = (1.0*one(T1), 1.0*one(T2))
#                P2o = (2.0*one(T1), 3.0*one(T2))
#                P2m = (2.0*one(T1), 2.0*one(T2))
#                P2u = (2.0*one(T1), 1.0*one(T2))
#                P3 = (3.0*one(T1), 3.0*one(T2))
#                @inferred ConvexDamage.is_convex(P1,P2m,P3)
#                @test ConvexDamage.is_convex(P1,P2o,P3)==false # non convex
#                @test ConvexDamage.is_convex(P1,P2m,P3)==true # convex (line --> not strictly)
#                @test ConvexDamage.is_convex(P1,P2u,P3)==true # strictly convex
#            end
#        end
#    end
#    @testset "combine()" begin
#        for material in materialvec
#            buf = ConvexDamage.build_buffer(material.convexstrategy)
#            state = ConvexDamage.init_materialstate(material)
#            state.convexificationbuffer.basebuffer.values .= [ConvexDamage.W_energy(x, material, state.damage) for x in state.convexificationbuffer.basebuffer.grid]
#            state.convexificationbuffer.basegrid_∂²W .= [Tensors.hessian(i->ConvexDamage.W_energy(i,material,state.damage), x) for x in state.convexificationbuffer.basebuffer.grid]
#            F_slp = getindex.(ConvexDamage.check_slope(state.convexificationbuffer.basebuffer.grid, state.convexificationbuffer.basebuffer.values),1)
#            F_hes = getindex.(ConvexDamage.check_hessian(state.convexificationbuffer.basegrid_∂²W, state.convexificationbuffer.basebuffer.grid, material.convexstrategy),1)
#            for T in [Float64, Tensors.Tensor{2,1,Float64,1}, Tensors.Tensor{4,1,Float64,1}]
#                F_s = ones(T,length(F_slp)).*F_slp
#                # F_hes hält werte
#                F_h = ones(T,length(F_hes)).*F_hes
#                Fpm = @inferred ConvexDamage.combine(F_s,F_h,0.4)
#                for (i,f) in enumerate([0.001,0.24589,0.7356,1.2254,14.6948,20.001])
#                    @test isapprox(Fpm[i],f*one(T); atol=1e-4)
#                end
#                # F_hes ist leer
#                F_h = Vector{T}()
#                Fpm = ConvexDamage.combine(F_s,F_h)
#                for (i,f) in enumerate([0.001,1.2254,14.6948,20.001])
#                    @test isapprox(Fpm[i],f*one(T); atol=1e-4)
#                end
#            end
#        end
#    end
#
#    @testset "discretize_interval()"  begin
#        for material in materialvec
#            buf = ConvexDamage.build_buffer(material.convexstrategy)
#            state = ConvexDamage.init_materialstate(material)
#            state.convexificationbuffer.basebuffer.values .= [ConvexDamage.W_energy(x, material, state.damage) for x in state.convexificationbuffer.basebuffer.grid]
#            state.convexificationbuffer.basegrid_∂²W .= [Tensors.hessian(i->ConvexDamage.W_energy(i,material,state.damage), x) for x in state.convexificationbuffer.basebuffer.grid]
#            F_slp = getindex.(ConvexDamage.check_slope(state.convexificationbuffer.basebuffer.grid, state.convexificationbuffer.basebuffer.values),1)
#            F_hes = getindex.(ConvexDamage.check_hessian(state.convexificationbuffer.basegrid_∂²W, state.convexificationbuffer.basebuffer.grid, material.convexstrategy),1)
#            for T in [Float64, Tensors.Tensor{2,1,Float64,1}]
#                F_s = ones(T,length(F_slp)).*F_slp
#                F_h = ones(T,length(F_hes)).*F_hes
#                Fpm = ConvexDamage.combine(F_s,F_h)
#                Fret = ones(T,length(buf.adaptivebuffer.grid)).*getindex.(buf.adaptivebuffer.grid,1)
#                @inferred ConvexDamage.discretize_interval(Fret, Fpm, ac)
#                @test isapprox(norm(getindex.(Fret,1)),119.80529; atol=1e-5)
#                # ================= helper fcns =================
#                # project() and Polynomial()
#                P = @inferred ConvexDamage.Polynomial(Fpm[4], Fpm[5]-Fpm[4], 30, material.convexstrategy)
#                @inferred project(P,0)
#                @test isapprox(project(P,0), Fpm[4]; atol=1e-4)
#                @test isapprox(project(P,30), Fpm[5]; atol=1e-4)
#                @test isapprox(project(P,15), (Fpm[4]+Fpm[5])/2; atol=1e-4)
#                Fadap = getindex.(map(x->project(P,x),collect(0:30)),1)
#                for i in 1:30 # check monotonicity of grid
#                    @test Fadap[i]<Fadap[i+1]
#                end
#                # distribute_gridpoints()
#                pntsperinterval = Vector{Int}(zeros(length(Fpm)-1))
#                @inferred ConvexDamage.distribute_gridpoints!(pntsperinterval, Fpm, ac)
#                @test sum(pntsperinterval)==ac.adaptivegrid_numpoints-1
#                for (i,n) in enumerate([8 16 16 40 34])
#                    @test pntsperinterval[i]===n
#                end
#            end
#            # inv_m()
#            mask = Vector{Bool}([0,1,0,1,0,0,0,0,1,1,1,0,1])
#            inv_mask = @inferred ConvexDamage.inv_m(mask)
#            for i in 1:length(mask)
#                @test mask[i]!=inv_mask[i]
#            end
#        end
#    end
#end