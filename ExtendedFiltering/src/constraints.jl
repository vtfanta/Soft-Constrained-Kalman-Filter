function inverse_distance_weight(dists::Vector{T}) where T
    w = zeros(size(dists))
    if any(dists .≈ 0)
        w[dists .≈ 0] .= 1
        return w
    else
        w .= 1 ./ dists
        w ./= sum(w)
        return w
    end
end

function get_track_constraint(se::StateEstimate, tracktree::KDTree, track::Matrix; 
    radius = 15, point_covariance = diagm([3,3]), minimal_no_nerby_points = 2)
    prev_state = se.x̂

    position = prev_state[[1,3]]

    nearby_points_idx = inrange(tracktree, position, radius, true)
    if length(nearby_points_idx) ≥ minimal_no_nerby_points
        dists = [norm(position - track[:,k]) for k in nearby_points_idx]
    else    # not enough points found inside given radius
        nearby_points_idx, dists = knn(tracktree, position, minimal_no_nerby_points, true)
    end
    weights = inverse_distance_weight(dists)
    nearby_points = [track[:,k] for k in nearby_points_idx] |> stack

    newmean = nearby_points * weights
    newcov = sum(weights[k] * ((point - newmean)*(point - newmean)' +
        point_covariance) for (k, point) in enumerate(eachcol(nearby_points)))
    
    newmean, newcov
end
