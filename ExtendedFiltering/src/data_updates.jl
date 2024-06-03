# Plain Kalman filter data update
function data_update_kf(se::StateEstimate{T}, meas_srcs) where T
    Σ = BlockDiagonals.BlockDiagonal([m.Σ for m in meas_srcs])
    H = vcat([m.∇output_fun(se) for m in meas_srcs]...)
    h = vcat([m.output_fun(se) for m in meas_srcs]...)
    z = vcat([m.z for m in meas_srcs]...)

    K = se.P * H' / (H * se.P * H' + Σ)
    x̂ₖ₊₁ = se.x̂ + K * (z - H * se.x̂)
    Pₖ₊₁ = (I(length(se.x̂)) - K * H) * se.P
    StateEstimate(x̂ₖ₊₁, Pₖ₊₁)
end

# Iterated extended Kalman filter Gauss-Newton data update
function data_update_iekf_gn(se0::StateEstimate{S}, meas_srcs,
    ϵ = 1e-3, maxiter = 30) where {S<:Real}
    function step(x̂ᵢ::Vector, se0, meas_srcs)
        Σ = BlockDiagonals.BlockDiagonal([m.Σ for m in meas_srcs])
        H = vcat([m.∇output_fun(se0) for m in meas_srcs]...)
        h = vcat([m.output_fun(se0) for m in meas_srcs]...)
        z = vcat([m.z for m in meas_srcs]...)

        K = se0.P * H' / (H * se0.P * H' + Σ)

        Δ = se0.x̂ - x̂ᵢ + K * (z - h - H * (se0.x̂ - x̂ᵢ))
        x̂ᵢ₊₁ = x̂ᵢ + Δ
        Pᵢ₊₁ = (I(length(x̂ᵢ)) - K * H) * se0.P
        StateEstimate(x̂ᵢ₊₁, Pᵢ₊₁)
    end

    iter = 1
    prev_est = se0
    next_est = nothing
    while iter < maxiter
        next_est = step(prev_est.x̂, se0, meas_srcs)
        iter += 1
        if LinearAlgebra.norm(next_est.x̂ - prev_est.x̂) < ϵ
            break
        end
        prev_est = next_est
    end
    next_est
end

# Iterated extended Kalman filter Gauss-Newton data update with step control
function data_update_iekf_gn_stepcontrol(se::StateEstimate{T}, meas_srcs;
    ϵ = 1e-3, maxiter = 30, maxstep = 0.1) where T

    function step(x̂ᵢ::Vector, se, meas_srcs)
        function V(x::Vector, se::StateEstimate{T}, meas_srcs)
            Σ = BlockDiagonals.BlockDiagonal([m.Σ for m in meas_srcs])
            h = vcat([m.output_fun(StateEstimate(x, se.P)) for m in meas_srcs]...)
            z = vcat([m.z for m in meas_srcs]...)

            first_term = z - h
            second_term = se.x̂ - x

            first_term' / Σ * first_term + second_term' / se.P * second_term
        end

        Σ = BlockDiagonals.BlockDiagonal([m.Σ for m in meas_srcs])
        H = vcat([m.∇output_fun(se) for m in meas_srcs]...)
        h = vcat([m.output_fun(se) for m in meas_srcs]...)
        z = vcat([m.z for m in meas_srcs]...)

        K = se.P * H' / (H * se.P * H' + Σ)

        Δ = se.x̂ - x̂ᵢ + K * (z - h - H * (se.x̂ - x̂ᵢ))

        V(α) = V(x̂ᵢ + α * Δ, se, meas_srcs)

        stepsize = Optim.minimizer(Optim.optimize(V, 0, 1, GoldenSection(),
        abs_tol = 1e-3, maxevals = 30))
        
        x̂ᵢ₊₁ = x̂ᵢ + stepsize * Δ

        Pᵢ₊₁ = (I(length(x̂ᵢ)) - K * H) * se.P
        StateEstimate(x̂ᵢ₊₁, Pᵢ₊₁)
    end

    iter = 1
    prev_est = se
    next_est = nothing
    while iter < maxiter
        next_est = step(prev_est.x̂, se, meas_srcs)
        iter += 1
        if LinearAlgebra.norm(next_est.x̂ - prev_est.x̂) < ϵ
            break
        end
        prev_est = next_est
    end
    next_est
end

# Iterated extended Kalman filter Levenberg-Marquardt data update
function data_update_iekf_lm(se0::StateEstimate{S}, meas_srcs, λ::T=1e-2,
    ϵ = 1e-3, maxiter = 30) where {S<:Real,T<:Real}

    function step(x̂ᵢ::Vector, se0, meas_srcs, λ)
        Σ = BlockDiagonals.BlockDiagonal([m.Σ for m in meas_srcs])
        H = vcat([m.∇output_fun(se0) for m in meas_srcs]...)
        h = vcat([m.output_fun(se0) for m in meas_srcs]...)
        z = vcat([m.z for m in meas_srcs]...)
        
        B = Diagonal(H' / Σ * H + inv(se0.P))
        P̃ = (I(length(x̂ᵢ)) - se0.P * (se0.P + 1/λ * inv(B))^(-1)) \ se0.P
        K = P̃ * H' / (H * P̃ * H' + Σ)

        Δ = se0.x̂ - x̂ᵢ + K * (z - h - H * (se0.x̂ - x̂ᵢ)) - λ * (I(length(x̂ᵢ)) - K * H) * P̃ * B * (se0.x̂ - x̂ᵢ)

        x̂ᵢ₊₁ = x̂ᵢ + Δ
        Pᵢ₊₁ = (I(length(x̂ᵢ)) - K * H) * se0.P
        StateEstimate(x̂ᵢ₊₁, Pᵢ₊₁)
    end

    iter = 1
    prev_est = se0
    next_est = nothing
    while iter < maxiter
        next_est = step(prev_est.x̂, se0, meas_srcs, λ)
        iter += 1
        if LinearAlgebra.norm(next_est.x̂ - prev_est.x̂) < ϵ
            break
        end
        prev_est = next_est
    end
    next_est
end

# Iterated extended Kalman filter Levenberg-Marquardt data update with step control
function data_update_iekf_lm_stepcontrol(se0::StateEstimate{S}, meas_srcs, λ::T=1e-2,
    ϵ = 1e-3, maxiter = 30) where {S<:Real,T<:Real}

    function step(x̂ᵢ::Vector, se0, meas_srcs, λ)
        function V(x::Vector, se::StateEstimate{T}, meas_srcs)
            Σ = BlockDiagonals.BlockDiagonal([m.Σ for m in meas_srcs])
            h = vcat([m.output_fun(StateEstimate(x, se0.P)) for m in meas_srcs]...)
            z = vcat([m.z for m in meas_srcs]...)

            first_term = z - h
            second_term = se.x̂ - x

            first_term' / Σ * first_term + second_term' / se.P * second_term
        end

        Σ = BlockDiagonals.BlockDiagonal([m.Σ for m in meas_srcs])
        H = vcat([m.∇output_fun(se0) for m in meas_srcs]...)
        h = vcat([m.output_fun(se0) for m in meas_srcs]...)
        z = vcat([m.z for m in meas_srcs]...)
        
        B = Diagonal(H' / Σ * H + inv(se0.P))
        P̃ = (I(length(x̂ᵢ)) - se0.P * (se0.P + 1/λ * inv(B))^(-1)) \ se0.P
        K = P̃ * H' / (H * P̃ * H' + Σ)

        Δ = se0.x̂ - x̂ᵢ + K * (z - h - H * (se0.x̂ - x̂ᵢ)) - λ * (I(length(x̂ᵢ)) - K * H) * P̃ * B * (se0.x̂ - x̂ᵢ)

        V(α) = V(x̂ᵢ + α * Δ, se0, meas_srcs)

        stepsize = Optim.minimizer(Optim.optimize(V, 0, 1, GoldenSection(), abs_tol = 1e-3, maxevals = 30))

        x̂ᵢ₊₁ = x̂ᵢ + stepsize * Δ
        Pᵢ₊₁ = (I(length(x̂ᵢ)) - K * H) * se0.P
        StateEstimate(x̂ᵢ₊₁, Pᵢ₊₁)
    end

    iter = 1
    prev_est = se0
    next_est = nothing
    while iter < maxiter
        next_est = step(prev_est.x̂, se0, meas_srcs, λ)
        iter += 1
        if LinearAlgebra.norm(next_est.x̂ - prev_est.x̂) < ϵ
            break
        end
        prev_est = next_est
    end
    next_est
end