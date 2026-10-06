#pragma once

#include <vector>
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <cmath>

#include "Robots.hh"
#include "src/collision/environment.hh"
#include "pop_settings.hh"

template <typename Robot>
struct PlannerResult {
    bool solved = false;
    std::vector<typename Robot::Configuration> path;
    std::vector<typename Robot::Configuration> simplified_path;
    // Every improving solution the optimizing planner produced along the way,
    // in the order they were found (cost is monotonically non-increasing).
    // Empty unless the planner runs >1 rrtc_iter. Used to export multiple seeds
    // (shorter = better) for a downstream trajectory optimizer.
    std::vector<std::pair<float, std::vector<typename Robot::Configuration>>> seed_paths;
    int start_tree_size = 0;
    int goal_tree_size = 0;
    int path_length = 0;
    int iters = 0;
    float cost = 0.0;
    std::size_t wall_ns = 0;   // wall time of the solve function
    std::size_t kernel_ns = 0; // just kernel runtime
    std::size_t copy_ns = 0;   // time to copy start/goals to gpu and copy path and path size back
};

template <typename Robot>
inline void print_cfg(typename Robot::Configuration &config) {
    for (int i = 0; i < Robot::dimension; i++) {
        std::cout << config[i] << " ";
    }
    std::cout << "\n";
}

template <typename Robot>
inline void print_cfg_to_ss(typename Robot::Configuration &config, std::stringstream &out) {
    for (int i = 0; i < Robot::dimension; i++) {
        out << config[i] << " ";
    }
    out << "\\n";
}

inline std::size_t
get_elapsed_nanoseconds(const std::chrono::time_point<std::chrono::steady_clock> &start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                start)
        .count();
}

// Host declarations for the CUDA planner.
namespace dRRT {
    template <typename Robot>
    PlannerResult<Robot>
    solve(typename Robot::Configuration &start, std::vector<typename Robot::Configuration> &goals,
          ppln::collision::Environment<float> &environment, pop_settings &settings,
          const int num_robots, const int block_per_robot);
}
