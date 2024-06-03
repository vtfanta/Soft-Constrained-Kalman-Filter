# Type definitions

mutable struct MeasurementSource{T<:Real,F<:Function,∇F<:Function}
    z::Vector{T}
    Σ::Matrix{T}
    output_fun::F
    ∇output_fun::∇F
end

mutable struct StateEstimate{T<:Real}
    x̂::Vector{T}
    P::Matrix{T}
end

mutable struct StateEquation{F<:Function,G<:Function,T<:Real}
    state_function::F
    ∇state_function::G
    Σ_process::Matrix{T}
end