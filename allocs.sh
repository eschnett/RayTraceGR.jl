#!/bin/bash

$HOME/julia-1.5/bin/julia --project=@. --track-allocation=user -e '
using Profile
using RayTraceGR
@time RayTraceGR.example2()
Profile.clear_malloc_data()
@time RayTraceGR.example2()
'
