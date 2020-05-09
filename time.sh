#!/bin/bash

$HOME/julia-1.5/bin/julia --project=@. -e '
using RayTraceGR
@time RayTraceGR.example2()
@time RayTraceGR.example2()
'
