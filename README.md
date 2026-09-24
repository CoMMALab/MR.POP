# MR. POP: MR. POP: Multi-Robot Parallel Optimizing Planner for Almost-Surely Asymptotically Optimal Planning

This README is still under construction, do not worry we'll be back
This repo holds the code for the paper "MR. POP: Multi-Robot Parallel Optimizing Planner for Almost-Surely Asymptotically Optimal Planning." 

## Building Code
To build MR. POP, follow the instructions below
```
git clone git@github.com:CoMMALab/MR.POP.git
cmake -B build
cmake --build build
```

## Running Benchmarks
MR. POP currently has two sets of benchmarks. The first is a set of 50 bin packing problems with four Frankas in the scene, totaling 28 degrees of freedom. The second is a set of 50 shelf reaching problems with five Frankas in the scene, totaling 35 degrees of freedom. The problems were based on the xECBS benchmark.
