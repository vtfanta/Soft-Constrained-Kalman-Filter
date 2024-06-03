using BasicInterpolators
using BlockDiagonals
using ExtendedFiltering
using JLD
using LinearAlgebra
using NearestNeighbors
using Plots
using ProgressMeter
using StatsPlots

## Load a track
track = JLD.load("./tracks/prague_line7.jld")["track"]

# Inspect the track
plot(track[1,:], track[2,:], label="Track", xlabel="UTM E [m]", ylabel="UTM N [m]")
savefig("figs/track.png")

# Build a KDTree for efficient nearest waypoint searches
tracktree = KDTree(track)

## Generate artificial distance and speed trajectories along the track
sampling_period = 0.1 # seconds

speed = vcat(
    repeat([0.], 100),
    range(start = 0., stop = 12., length = 150),
    repeat([12.], 100),
    range(start = 12., stop = 0., length = 150),
    repeat([0.], 100)
)

distance = cumsum(speed) * sampling_period

time = collect(range(start = 0., step = sampling_period, length = length(speed)))

## Find 2D coordinates of the trajectory at the given distances by interpolation
track_length = sum(diff(track, dims=2).^2, dims=1) .|> sqrt |> vec |> cumsum
track_length = [0.; track_length]
itpx = LinearInterpolator(track_length, track[1,:])
itpy = LinearInterpolator(track_length, track[2,:])

x = itpx.(distance)
y = itpy.(distance)

## Create artificial GNSS measurements by adding noise to the clean data
x_noisy = copy(x)
x_noisy[50:200] -= 5. .+ randn(151)
x_noisy[250:400] -= 5. .+ randn(151)
x_noisy[450:550] += 2. * randn(101)
y_noisy = copy(y)
y_noisy[50:150] += 2. .+ randn(101)
y_noisy[200:350] -= 5. .+ randn(151)
y_noisy[400:550] += 3. * randn(151)

speed_noisy = speed .+ 0.5 * randn(length(speed))

# Compare with the clean data
plot(x, y, label="Clean trajectory", xlabel="UTM E [m]", ylabel="UTM N [m]")
plot!(x_noisy, y_noisy, label="Noisy trajectory")
savefig("figs/noisy_position.png")

plot(time, speed, label="Clean speed", xlabel="Time [s]", ylabel="Speed [m/s]")
plot!(time, speed_noisy, label="Noisy speed")
savefig("figs/noisy_speed.png")

## Set up the iterated extended Kalman filter for constant velocity model
# state vector: [x, vx, y, vy]

# Process noise covariance matrix
Σ_process = 1e-2 * diagm([1., 1., 1., 1.])

# Define the state equation
transition_matrix_1D = [1. sampling_period; 0. 1.]
transition_matrix_2D = BlockDiagonal([transition_matrix_1D, transition_matrix_1D])
state_function(x̂::StateEstimate) = transition_matrix_2D * x̂.x̂
∇state_function(x̂::StateEstimate) = transition_matrix_2D'
state_equation = StateEquation(state_function, ∇state_function, Σ_process)


# Measurement noise covariance matrix
# measurement vector: [x, y, √(vx^2 + vy^2)]
Σ_measurement = diagm([3., 3., 1.])

# Define the GNSS output equation
output_function_GNSS(x̂::StateEstimate) = [x̂.x̂[1]; x̂.x̂[3]; norm(x̂.x̂[[2,4]])]
function ∇output_function_GNSS(x̂::StateEstimate)
    denom = max(norm(x̂.x̂[[2,4]]), 1e-2) # to avoid division by near-zero
    return [
        1. 0. 0. 0.;
        0. 0. 1. 0.;
        0. x̂.x̂[2] / denom 0. x̂.x̂[4] / denom
    ]
end

# Define the track soft-constraint pseudomeasurement
Σ_trackpoint = diagm([1., 1.])

output_function_track(x̂::StateEstimate) = [x̂.x̂[1], x̂.x̂[3]]
∇output_function_track(x̂::StateEstimate) = [1. 0. 0. 0.; 0. 0. 1. 0.]

## Estimation
# Initial state estimate
state_est = StateEstimate([x_noisy[1]; 0.; y_noisy[1]; 0.], diagm([1., 1., 1., 1.]))

# Save output
state_estimates = Vector{StateEstimate}(undef, length(time))
GNSS_measurements = Vector{MeasurementSource}(undef, length(time))
track_pseudomeas = Vector{MeasurementSource}(undef, length(time))

# Estimation loop
for (k, t) in enumerate(time)
    # Filtering step
    z_GNSS = [x_noisy[k], y_noisy[k], speed_noisy[k]]
    GNSS_meas = MeasurementSource(z_GNSS, Σ_measurement,
        output_function_GNSS, ∇output_function_GNSS)

    z_track, Σ_track = get_track_constraint(state_est, tracktree, track,
        point_covariance=Σ_trackpoint)
    track_meas = MeasurementSource(z_track, Σ_track,
        output_function_track, ∇output_function_track)

    global state_est = data_update_iekf_gn_stepcontrol(state_est, [GNSS_meas, track_meas])

    # Save the state estimate
    state_estimates[k] = state_est
    GNSS_measurements[k] = GNSS_meas
    track_pseudomeas[k] = track_meas

    # Prediction step
    state_est = prediction_kf(state_est, state_equation)
end

## Plot the results
x_pred = [x̂.x̂[1] for x̂ in state_estimates]
vx_pred = [x̂.x̂[2] for x̂ in state_estimates]
y_pred = [x̂.x̂[3] for x̂ in state_estimates]
vy_pred = [x̂.x̂[4] for x̂ in state_estimates]
speed_pred = [norm([vx, vy]) for (vx, vy) in zip(vx_pred, vy_pred)]

plot(x, y, label="True trajectory", xlabel="UTM E [m]", ylabel="UTM N [m]")
plot!(x_noisy, y_noisy, label="Noisy trajectory")
plot!(x_pred, y_pred, label="Estimated trajectory")
savefig("figs/predicted_position.png")

plot(time, speed, label="True speed", xlabel="Time [s]", ylabel="Speed [m/s]")
plot!(time, speed_noisy, label="Noisy speed")
plot!(time, speed_pred, label="Estimated speed")
savefig("figs/predicted_speed.png")

## Animation
p = Progress(length(time), desc="Animating...")
anim = @animate for k in 1:length(time)
    next!(p)
    plot(track[1,:], track[2,:], marker=:x, label="Track", xlabel="UTM E [m]", ylabel="UTM N [m]")
    covellipse!(state_estimates[k].x̂[[1,3]], state_estimates[k].P[[1,3],[1,3]], label="Estimated position")
    scatter!([x[k]], [y[k]], label="True position", marker=:o)
    xlims!(x[k] - 20, x[k] + 20)
    ylims!(y[k] - 20, y[k] + 20)
    covellipse!(track_pseudomeas[k].z, track_pseudomeas[k].Σ, label="Track constraint")
    covellipse!(GNSS_measurements[k].z[1:2], GNSS_measurements[k].Σ[[1,2],[1,2]], label="GNSS measurement")
end
gif(anim, "figs/soft_constraint_example.gif", fps=1/sampling_period |> Int)