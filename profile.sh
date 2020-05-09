#!/bin/bash

$HOME/julia-1.5/bin/julia --project=@. -e '
using Profile
using RayTraceGR
@profile RayTraceGR.example2()
Profile.clear()
@profile RayTraceGR.example2()
Profile.print()
' | tee profile.out
