# Soft-Constrained Kalman Filter
This tutorial describes a way to introduce soft constraints into an iterated
Kalman filter scheme. 
The code is written in the [Julia](https://julialang.org/) programming language and
shows the application of soft constraints on the example of tram localization.

## Environment Setup
After downloading the repository, start a terminal in the repo's root directory
and start Julia. Afterwards, type
```julia
using Pkg
Pkg.instantiate()
```
This will install all the packages needed for this tutorial.

## Data Preparation
Since the motion of a tram is inherently restricted to the tramway tracks,
it makes sense for a multi-dimensional Kalman filter to acknowledge this fact.
For the purpose of this tutorial, we will generate artificial tram trajectory along
the track of line 7 in Prague. The waypoints of the track are saved in a JLD [file](tracks/prague_line7.jld).
The track can be loaded with
```julia
using JLD
track = JLD.load("tracks/prague_line7.jld")["track"]
```
Afterwards, let's construct a [k-d tree](https://www.wikiwand.com/en/K-d_tree) for
fast look-up of nearest waypoints with the help of the `NearestNeighbors.jl` package.
```julia
using NearestNeighbors
tracktree = KDTree(track)
```

![Track](figs/track.png)

```julia
using ExtendedFiltering
```

![Noisy position readings](figs/noisy_position.png)

![Noisy speed readings](figs/noisy_speed.png)

![Estimated position](figs/predicted_position.png)

![Estimated speed](figs/predicted_speed.png)

![Animation of the estimation procedure](figs/soft_constraint_example.gif)