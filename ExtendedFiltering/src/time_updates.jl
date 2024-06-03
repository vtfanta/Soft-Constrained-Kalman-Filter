# Kalman filter prediction step
function prediction_kf(se::StateEstimate, state_equation::StateEquation)
    F = state_equation.∇state_function(se)
    x̂₋ = state_equation.state_function(se)
    P₋ = F * se.P * F' + state_equation.Σ_process
    StateEstimate(x̂₋, P₋)
end