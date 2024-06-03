function simulate_iekf(
    df::DataFrame,
    T::Vector{Float64},
    Σ_GNSS::Matrix{Float64},
    Σ_trackpoint::Matrix{Float64},
    trackpoint_radius::Float64,
    σ_direction::Float64,
    x̂0::Vector{Float64},
    P0::Matrix,
    output_fun_GNSS::Function,
    ∇output_fun_GNSS::Function,
    output_fun_track::Function,
    ∇output_fun_track::Function,
    state_eq::StateEquation,
    track::Matrix{Float64},
    tracktree::KDTree,
    data_update_fun::Function
)

    se = StateEstimate(x̂0, P0)
    state_estimates = Vector{StateEstimate{Float64}}(undef, length(T))
    measurements_GNSS = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_track = Vector{MeasurementSource{Float64}}(undef, length(T))
    for (k, t) in enumerate(T)
        # Data update
        z_GNSS = [df.utmx_meas[k], df.utmy_meas[k], df.speed_meas[k]]
        meas_GNSS = MeasurementSource(z_GNSS, Σ_GNSS, output_fun_GNSS, ∇output_fun_GNSS)

        z_track, Σ_track, a = get_track_constraint(se, tracktree, track, point_covariance = Σ_trackpoint, radius = trackpoint_radius)
        z_track = vcat(z_track, 0)
        meas_track = MeasurementSource(z_track, BlockDiagonals.BlockDiagonal([Σ_track, diagm([σ_direction])]) |> Matrix,
            s->output_fun_track(a, s), s->∇output_fun_track(a, s))

        se = data_update_fun(se, [meas_GNSS, meas_track])

        # Store results
        state_estimates[k] = se
        measurements_GNSS[k] = meas_GNSS
        measurements_track[k] = meas_track

        # State update
        se = prediction_kf(se, state_eq)
    end

    dfret = DataFrame(
        T = T,
        x_pred = [se.x̂[1] for se in state_estimates],
        vx_pred = [se.x̂[2] for se in state_estimates],
        y_pred = [se.x̂[3] for se in state_estimates],
        vy_pred = [se.x̂[4] for se in state_estimates],
        P = [se.P for se in state_estimates],
        meas_GNSS = measurements_GNSS,
        meas_track = measurements_track,
        meanx_track = [m.z[1] for m in measurements_track],
        meany_track = [m.z[2] for m in measurements_track],
        P_track = [m.Σ for m in measurements_track],
        utmx_KF = df.utmx_KF,
        utmy_KF = df.utmy_KF
    )
    return dfret
end

function simulate_CAM_iekf(
    df::DataFrame,
    T::Vector{Float64},
    Σ_GNSS::Matrix{Float64},
    Σ_IMU::Matrix{Float64},
    Σ_trackpoint::Matrix{Float64},
    waypoint_radius::Float64,
    σ_direction::Float64,
    x̂0::Vector{Float64},
    P0::Matrix,
    output_fun_GNSS::Function,
    ∇output_fun_GNSS::Function,
    output_fun_IMU::Function,
    ∇output_fun_IMU::Function,
    output_fun_track::Function,
    ∇output_fun_track::Function,
    state_eq::StateEquation,
    track::Matrix{Float64},
    tracktree::KDTree,
    data_update_fun::Function
)

    se = StateEstimate(x̂0, P0)
    state_estimates = Vector{StateEstimate{Float64}}(undef, length(T))
    measurements_GNSS = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_IMU = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_track = Vector{MeasurementSource{Float64}}(undef, length(T))
    for (k, t) in enumerate(T)
        # Data update
        z_GNSS = [df.utmx_meas[k], df.utmy_meas[k], df.speed_meas[k]]
        meas_GNSS = MeasurementSource(z_GNSS, Σ_GNSS, output_fun_GNSS, ∇output_fun_GNSS)

        z_IMU = [df.acc_filt_forward[k], df.acc_filt_right[k]]
        meas_IMU = MeasurementSource(z_IMU, Σ_IMU, output_fun_IMU, ∇output_fun_IMU)

        z_track, Σ_track, a = get_track_constraint(StateEstimate(se.x̂[[1,2,4,5]], se.P[[1,2,4,5],[1,2,4,5]]), tracktree, track, point_covariance = Σ_trackpoint,
            radius = waypoint_radius)
        z_track = vcat(z_track, 0)
        meas_track = MeasurementSource(z_track, BlockDiagonals.BlockDiagonal([Σ_track, diagm([σ_direction])]) |> Matrix,
            s->output_fun_track(a, s), s->∇output_fun_track(a, s))

        se = data_update_fun(se, [meas_GNSS, meas_IMU, meas_track])

        # Store results
        state_estimates[k] = se
        measurements_GNSS[k] = meas_GNSS
        measurements_IMU[k] = meas_IMU
        measurements_track[k] = meas_track

        # State update
        se = prediction_kf(se, state_eq)
    end

    dfret = DataFrame(
        T = T,
        x_pred = [se.x̂[1] for se in state_estimates],
        vx_pred = [se.x̂[2] for se in state_estimates],
        ax_pred = [se.x̂[3] for se in state_estimates],
        y_pred = [se.x̂[4] for se in state_estimates],
        vy_pred = [se.x̂[5] for se in state_estimates],
        ay_pred = [se.x̂[6] for se in state_estimates],
        θ_pred = [se.x̂[7] for se in state_estimates],
        P = [se.P for se in state_estimates],
        meas_GNSS = measurements_GNSS,
        meas_track = measurements_track,
        meanx_track = [m.z[1] for m in measurements_track],
        meany_track = [m.z[2] for m in measurements_track],
        P_track = [m.Σ for m in measurements_track],
        utmx_KF = df.utmx_KF,
        utmy_KF = df.utmy_KF
    )
    return dfret
end

function simulate_CAMv2_iekf(
    df::DataFrame,
    T::Vector{Float64},
    Σ_GNSS::Matrix{Float64},
    Σ_IMU::Matrix{Float64},
    Σ_trackpoint::Matrix{Float64},
    waypoint_radius::Float64,
    σ_direction::Float64,
    x̂0::Vector{Float64},
    P0::Matrix,
    output_fun_GNSS::Function,
    ∇output_fun_GNSS::Function,
    output_fun_IMU::Function,
    ∇output_fun_IMU::Function,
    output_fun_track::Function,
    ∇output_fun_track::Function,
    state_eq::StateEquation,
    track::Matrix{Float64},
    tracktree::KDTree,
    data_update_fun::Function
)

    se = StateEstimate(x̂0, P0)
    state_estimates = Vector{StateEstimate{Float64}}(undef, length(T))
    measurements_GNSS = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_IMU = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_track = Vector{MeasurementSource{Float64}}(undef, length(T))
    for (k, t) in enumerate(T)
        # Data update
        z_GNSS = [df.utmx_meas[k], df.utmy_meas[k], df.speed_meas[k]]
        meas_GNSS = MeasurementSource(z_GNSS, Σ_GNSS, output_fun_GNSS, ∇output_fun_GNSS)

        z_track, Σ_track, a = get_track_constraint(StateEstimate(se.x̂[[1,2,4,5]], se.P[[1,2,4,5],[1,2,4,5]]), tracktree, track, point_covariance = Σ_trackpoint, 
            radius = waypoint_radius)
        z_track = vcat(z_track, 0)
        meas_track = MeasurementSource(z_track, BlockDiagonals.BlockDiagonal([Σ_track, diagm([σ_direction])]) |> Matrix,
            s->output_fun_track(a, s), s->∇output_fun_track(a, s))

        z_IMU = [df.acc_filt_forward[k], 0]
        meas_IMU = MeasurementSource(z_IMU, Σ_IMU, 
            s->output_fun_IMU(a, s), s->∇output_fun_IMU(a, s))

        se = data_update_fun(se, [meas_GNSS, meas_IMU, meas_track])

        # Store results
        state_estimates[k] = se
        measurements_GNSS[k] = meas_GNSS
        measurements_IMU[k] = meas_IMU
        measurements_track[k] = meas_track

        # State update
        se = prediction_kf(se, state_eq)
    end

    dfret = DataFrame(
        T = T,
        x_pred = [se.x̂[1] for se in state_estimates],
        vx_pred = [se.x̂[2] for se in state_estimates],
        ax_pred = [se.x̂[3] for se in state_estimates],
        y_pred = [se.x̂[4] for se in state_estimates],
        vy_pred = [se.x̂[5] for se in state_estimates],
        ay_pred = [se.x̂[6] for se in state_estimates],
        P = [se.P for se in state_estimates],
        meas_GNSS = measurements_GNSS,
        meas_track = measurements_track,
        meanx_track = [m.z[1] for m in measurements_track],
        meany_track = [m.z[2] for m in measurements_track],
        P_track = [m.Σ for m in measurements_track],
        utmx_KF = df.utmx_KF,
        utmy_KF = df.utmy_KF
    )
    return dfret
end

function simulate_CVM_curv_iekf(
    df::DataFrame,
    T::Vector{Float64},
    Σ_GNSS::Matrix{Float64},
    Σ_trackpoint::Matrix{Float64},
    waypoint_radius::Float64,
    σ_direction::Float64,
    x̂0::Vector{Float64},
    P0::Matrix,
    output_fun_GNSS::Function,
    ∇output_fun_GNSS::Function,
    output_fun_track::Function,
    ∇output_fun_track::Function,
    state_eq::StateEquation,
    track::Matrix{Float64},
    tracktree::KDTree,
    data_update_fun::Function,
    point_curvatures::Vector{Float64},
    curv_radius::Float64,
    curv_gyro_thrs::Float64 = 3.,
    curv_spd_thrs::Float64 = 1.
)

    se = StateEstimate(x̂0, P0)
    state_estimates = Vector{StateEstimate{Float64}}(undef, length(T))
    measurements_GNSS = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_track = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_curv = Vector{Union{MeasurementSource{Float64}, Nothing}}(undef, length(T))

    for (k, t) in enumerate(T)
        # Data update
        z_GNSS = [df.utmx_meas[k], df.utmy_meas[k], df.speed_meas[k]]
        meas_GNSS = MeasurementSource(z_GNSS, Σ_GNSS, output_fun_GNSS, ∇output_fun_GNSS)

        z_track, Σ_track, a = get_track_constraint(se, tracktree, track, point_covariance = Σ_trackpoint,
            radius = waypoint_radius)
        z_track = vcat(z_track, 0)
        meas_track = MeasurementSource(z_track, BlockDiagonals.BlockDiagonal([Σ_track, diagm([σ_direction])]) |> Matrix,
            s->output_fun_track(a, s), s->∇output_fun_track(a, s))
        
        should_calculate_curv = abs(df.rot_left[k]) > curv_gyro_thrs && (sum(se.x̂[[2,4]].^2)) > curv_spd_thrs

        if should_calculate_curv
            nearby_curv_points_idx = inrange(tracktree, se.x̂[[1, 3]], curv_radius)
            nearby_curvs = point_curvatures[nearby_curv_points_idx]
            current_curv = df.acc_filt_right[k] / sum(se.x̂[[2,4]].^2)
            diffs = abs.(nearby_curvs .- current_curv)
            curv_weights = inverse_distance_weight(diffs)
            nearby_curv_points = [track[:,k] for k in nearby_curv_points_idx] |> stack
            new_mean = nearby_curv_points * curv_weights
            new_cov = sum(curv_weights[k] * ((point - new_mean)*(point - new_mean)' + Σ_trackpoint) for (k, point) in enumerate(eachcol(nearby_curv_points)))
            z_curv = new_mean
            Σ_curv = new_cov
            meas_curv = MeasurementSource(z_curv, Σ_curv, s -> s.x̂[[1,3]], s -> [1 0 0 0;0 0 1 0])

            se = data_update_fun(se, [meas_GNSS, meas_track, meas_curv])
        else
            se = data_update_fun(se, [meas_GNSS, meas_track])
        end

        # Store results
        if should_calculate_curv
            state_estimates[k] = se
            measurements_GNSS[k] = meas_GNSS
            measurements_track[k] = meas_track
            measurements_curv[k] = meas_curv
        else
            state_estimates[k] = se
            measurements_GNSS[k] = meas_GNSS
            measurements_track[k] = meas_track
            measurements_curv[k] = nothing
        end

        # State update
        se = prediction_kf(se, state_eq)
    end

    dfret = DataFrame(
        T = T,
        x_pred = [se.x̂[1] for se in state_estimates],
        vx_pred = [se.x̂[2] for se in state_estimates],
        y_pred = [se.x̂[3] for se in state_estimates],
        vy_pred = [se.x̂[4] for se in state_estimates],
        P = [se.P for se in state_estimates],
        meas_GNSS = measurements_GNSS,
        meas_track = measurements_track,
        meas_curv = measurements_curv,
        meanx_track = [m.z[1] for m in measurements_track],
        meany_track = [m.z[2] for m in measurements_track],
        P_track = [m.Σ for m in measurements_track],
        utmx_KF = df.utmx_KF,
        utmy_KF = df.utmy_KF
    )
    return dfret
end

function simulate_CAM_curv_iekf(
    df::DataFrame,
    T::Vector{Float64},
    Σ_GNSS::Matrix{Float64},
    Σ_IMU::Matrix{Float64},
    Σ_trackpoint::Matrix{Float64},
    waypoint_radius::Float64,
    σ_direction::Float64,
    x̂0::Vector{Float64},
    P0::Matrix,
    output_fun_GNSS::Function,
    ∇output_fun_GNSS::Function,
    output_fun_IMU::Function,
    ∇output_fun_IMU::Function,
    output_fun_track::Function,
    ∇output_fun_track::Function,
    state_eq::StateEquation,
    track::Matrix{Float64},
    tracktree::KDTree,
    data_update_fun::Function,
    point_curvatures::Vector{Float64},
    curv_radius::Float64,
    curv_gyro_thrs::Float64 = 3.,
    curv_spd_thrs::Float64 = 1.
)

    se = StateEstimate(x̂0, P0)
    state_estimates = Vector{StateEstimate{Float64}}(undef, length(T))
    measurements_GNSS = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_IMU = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_track = Vector{MeasurementSource{Float64}}(undef, length(T))
    measurements_curv = Vector{Union{MeasurementSource{Float64}, Nothing}}(undef, length(T))

    for (k, t) in enumerate(T)
        # Data update
        z_GNSS = [df.utmx_meas[k], df.utmy_meas[k], df.speed_meas[k]]
        meas_GNSS = MeasurementSource(z_GNSS, Σ_GNSS, output_fun_GNSS, ∇output_fun_GNSS)

        z_track, Σ_track, a = get_track_constraint(StateEstimate(se.x̂[[1,2,4,5]], se.P[[1,2,4,5],[1,2,4,5]]), tracktree, track, point_covariance = Σ_trackpoint,
            radius = waypoint_radius)
        z_track = vcat(z_track, 0)
        meas_track = MeasurementSource(z_track, BlockDiagonals.BlockDiagonal([Σ_track, diagm([σ_direction])]) |> Matrix,
            s->output_fun_track(a, s), s->∇output_fun_track(a, s))

        z_IMU = [norm([df.acc_filt_forward[k],df.acc_filt_right[k]]), 0]
        meas_IMU = MeasurementSource(z_IMU, Σ_IMU, 
            s->output_fun_IMU(a, s), s->∇output_fun_IMU(a, s))

        should_calculate_curv = abs(df.rot_left[k]) > curv_gyro_thrs && (sum(se.x̂[[2,5]].^2)) > curv_spd_thrs

        if should_calculate_curv
            nearby_curv_points_idx = inrange(tracktree, se.x̂[[1, 4]], curv_radius)
            nearby_curvs = point_curvatures[nearby_curv_points_idx]
            current_curv = df.acc_filt_right[k] / sum(se.x̂[[2,5]].^2)
            diffs = abs.(nearby_curvs .- current_curv)
            curv_weights = inverse_distance_weight(diffs)
            nearby_curv_points = [track[:,k] for k in nearby_curv_points_idx] |> stack
            new_mean = nearby_curv_points * curv_weights
            new_cov = sum(curv_weights[k] * ((point - new_mean)*(point - new_mean)' + Σ_trackpoint) for (k, point) in enumerate(eachcol(nearby_curv_points)))
            z_curv = new_mean
            Σ_curv = new_cov
            meas_curv = MeasurementSource(z_curv, Σ_curv, s -> s.x̂[[1,4]], s -> [1 0 0 0 0 0;0 0 0 1 0 0])

            se = data_update_fun(se, [meas_GNSS, meas_track, meas_IMU, meas_curv])
        else
            se = data_update_fun(se, [meas_GNSS, meas_track, meas_IMU])
        end

        # Store results
        if should_calculate_curv
            state_estimates[k] = se
            measurements_GNSS[k] = meas_GNSS
            measurements_track[k] = meas_track
            measurements_curv[k] = meas_curv
        else
            state_estimates[k] = se
            measurements_GNSS[k] = meas_GNSS
            measurements_track[k] = meas_track
            measurements_curv[k] = nothing
        end
        # Store results
        state_estimates[k] = se
        measurements_GNSS[k] = meas_GNSS
        measurements_IMU[k] = meas_IMU
        measurements_track[k] = meas_track

        # State update
        se = prediction_kf(se, state_eq)
    end

    dfret = DataFrame(
        T = T,
        x_pred = [se.x̂[1] for se in state_estimates],
        vx_pred = [se.x̂[2] for se in state_estimates],
        ax_pred = [se.x̂[3] for se in state_estimates],
        y_pred = [se.x̂[4] for se in state_estimates],
        vy_pred = [se.x̂[5] for se in state_estimates],
        ay_pred = [se.x̂[6] for se in state_estimates],
        P = [se.P for se in state_estimates],
        meas_GNSS = measurements_GNSS,
        meas_track = measurements_track,
        meas_curv = measurements_curv,
        meanx_track = [m.z[1] for m in measurements_track],
        meany_track = [m.z[2] for m in measurements_track],
        P_track = [m.Σ for m in measurements_track],
        utmx_KF = df.utmx_KF,
        utmy_KF = df.utmy_KF
    )
    return dfret
end

function simulate_1D_CAM_kf(
    df::DataFrame,
    T::Vector{Float64},
    Σ_1D_GNSS_IMU::Matrix{Float64},
    x̂0::Vector{Float64},
    P0::Matrix,
    output_fun_1D_GNSS_IMU_CAM::Function,
    ∇output_fun_1D_GNSS_IMU::Function,
    state_eq::StateEquation,
    dst_along::Vector{Float64},
    data_update_fun::Function
)

    se = StateEstimate(x̂0, P0)
    state_estimates = Vector{StateEstimate{Float64}}(undef, length(T))
    measurements_GNSS = Vector{MeasurementSource{Float64}}(undef, length(T))
    for (k, t) in enumerate(T)
        # Data update
        z_GNSS_IMU = [dst_along[k], df.acc_filt_forward[k]]
        meas_GNSS_IMU = MeasurementSource(z_GNSS_IMU, Σ_1D_GNSS_IMU, output_fun_1D_GNSS_IMU_CAM, ∇output_fun_1D_GNSS_IMU)

        se = data_update_fun(se, [meas_GNSS_IMU])
        
        # Time update
        se = prediction_kf(se, state_eq)

        # Store results
        state_estimates[k] = se
        measurements_GNSS[k] = meas_GNSS_IMU
    end

    dfret = DataFrame(
        T = T,
        x_pred = [se.x̂[1] for se in state_estimates],
        v_pred = [se.x̂[2] for se in state_estimates],
        a_pred = [se.x̂[3] for se in state_estimates],
        P = [se.P for se in state_estimates],
        meas_GNSS = measurements_GNSS,
        utmx_KF = df.utmx_KF,
        utmy_KF = df.utmy_KF
    )
    return dfret
end

function simulate_1D_CVM_kf(
    df::DataFrame,
    T::Vector{Float64},
    Σ_1D_GNSS::Matrix,
    x̂0::Vector{Float64},
    P0::Matrix,
    output_fun_1D_GNSS_CVM::Function,
    ∇output_fun_1D_GNSS_CVM::Function,
    Σ_process_CVM::Matrix,
    dst_along::Vector{Float64},
    data_update_fun::Function
)

    se = StateEstimate(x̂0, P0)
    state_estimates = Vector{StateEstimate{Float64}}(undef, length(T))
    measurements_GNSS = Vector{MeasurementSource{Float64}}(undef, length(T))
    for (k, t) in enumerate(T)
        # Data update
        z_GNSS = [dst_along[k], df.speed[k]]
        meas_GNSS = MeasurementSource(z_GNSS, Σ_1D_GNSS, output_fun_1D_GNSS_CVM, ∇output_fun_1D_GNSS_CVM)

        se = data_update_fun(se, [meas_GNSS])

        # Time update with variable time
        Ts = k == length(T) ? 0.1 : T[k+1] - T[k]
        new_state_eq = StateEquation(
            se -> [1 Ts; 0 1] * se.x̂,
            se -> [1 Ts; 0 1]',
            Σ_process_CVM
        )
        se = prediction_kf(se, new_state_eq)

        # Store results
        state_estimates[k] = se
        measurements_GNSS[k] = meas_GNSS
    end
    dfret = DataFrame(
        T = T,
        x_pred = [se.x̂[1] for se in state_estimates],
        v_pred = [se.x̂[2] for se in state_estimates],
        P = [se.P for se in state_estimates],
        meas_GNSS = measurements_GNSS
    )
end