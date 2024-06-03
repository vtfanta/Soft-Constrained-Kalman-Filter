module ExtendedFiltering

using BlockDiagonals
using DataFrames
using DataToolkit
using LinearAlgebra
using NearestNeighbors
using Optim
using ProgressMeter
using StatsPlots

export MeasurementSource, StateEstimate, StateEquation
export prediction_kf
export data_update_kf
export data_update_iekf_gn, data_update_iekf_gn_stepcontrol
export data_update_iekf_lm, data_update_iekf_lm_stepcontrol
export inverse_distance_weight, get_track_constraint
export simulate_iekf, simulate_CAM_iekf, simulate_CAMv2_iekf, simulate_CVM_curv_iekf, simulate_CAM_curv_iekf, simulate_1D_CAM_kf
export simulate_1D_CVM_kf

include("types.jl")
include("time_updates.jl")
include("data_updates.jl")
include("constraints.jl")
include("simulation.jl")

end # module