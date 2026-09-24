// Run the pRRTC (AO-RRT-Connect) planner on the panda_four benchmark and export
// its collision-free geometric paths as plain-text seed files that cuRobo's
// trajectory optimizer can consume (see curobo/seed_from_path.py).
//
// This is the "generate paths" half of the asao -> cuRobo pipeline. It differs
// from evaluate_mr.cu in two ways:
//   (1) it writes one seed file per problem (28 joint values per line, radians)
//       instead of a benchmark CSV, and
//   (2) it runs the planner as an OPTIMIZING planner (rrtc_iter > 1) and, thanks
//       to PlannerResult::seed_paths, also dumps every improving solution it
//       found along the way -- so a downstream optimizer can pick among several
//       seeds, preferring the shorter ones.
//
// Output layout (out_dir):
//   <name>.txt                          final (best / shortest) path
//   <name>__k00__cost<c>.txt            initial solution      (rrtc_iter 0)
//   <name>__k01__cost<c>.txt            first improvement     ...
//   ...                                 (only when rrtc_iter > 1)
//
// Usage:
//   export_seeds <robot_name> <out_dir> [rrtc_iter] [optimize_iters]
//   e.g.  ./export_seeds panda_four ../curobo/seeds 6

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "src/collision/environment.hh"
#include "src/collision/factory.hh"
#include "src/planning/Planners.hh"
#include "src/planning/pRRTC_settings.hh"


using json = nlohmann::json;

using namespace ppln::collision;

// Identical obstacle construction to evaluate_mr.cu so the scene the planner
// sees matches the benchmark exactly.
Environment<float> problem_dict_to_env(const json& problem, const std::string& name) {
    Environment<float> env{};

    std::vector<Sphere<float>> spheres;
    std::vector<Capsule<float>> capsules;
    std::vector<Cuboid<float>> cuboids;
    for (const auto& obj : problem["sphere"]) {
        const json& position = obj["position"];
        Sphere<float> sphere(position[0], position[1], position[2], obj["radius"]);
        sphere.name = obj["name"];
        spheres.push_back(sphere);
    }
    if (name == "box") {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const std::array<float, 3> dims = {radius, radius, radius / 2.0f};
            auto cuboid = factory::cuboid::array(position, orientation, dims);
            cuboid.name = obj["name"];
            cuboids.push_back(cuboid);
        }
    } else {
        for (const auto& obj : problem["cylinder"]) {
            const json& position = obj["position"];
            const json& orientation = obj["orientation_euler_xyz"];
            const float radius = obj["radius"];
            const float length = obj["length"];
            auto cylinder = factory::cylinder::center::array(position, orientation, radius, length);
            cylinder.name = obj["name"];
            capsules.push_back(cylinder);
        }
    }
    for (const auto& obj : problem["box"]) {
        const json& position = obj["position"];
        const json& orientation = obj["orientation_euler_xyz"];
        const json& half_extents = obj["half_extents"];
        auto cuboid = factory::cuboid::array(position, orientation, half_extents);
        cuboid.name = obj["name"];
        cuboids.push_back(cuboid);
    }

    if (!spheres.empty()) {
        env.spheres = new Sphere<float>[spheres.size()];
        std::copy(spheres.begin(), spheres.end(), env.spheres);
        env.num_spheres = spheres.size();
    }
    if (!capsules.empty()) {
        env.capsules = new Capsule<float>[capsules.size()];
        std::copy(capsules.begin(), capsules.end(), env.capsules);
        env.num_capsules = capsules.size();
    }
    if (!cuboids.empty()) {
        env.cuboids = new Cuboid<float>[cuboids.size()];
        std::copy(cuboids.begin(), cuboids.end(), env.cuboids);
        env.num_cuboids = cuboids.size();
    }
    return env;
}

// Write a single geometric path (one waypoint per line, dim joint values per
// line, radians) in exactly the format curobo/seed_from_path.py::parse_path_text
// expects.
template <typename Robot>
void write_path_file(const std::string& fname,
                     const std::vector<typename Robot::Configuration>& path) {
    std::ofstream out(fname);
    out << std::setprecision(9);
    for (const auto& cfg : path) {
        for (int i = 0; i < Robot::dimension; i++) {
            out << cfg[i];
            if (i + 1 < Robot::dimension) out << ' ';
        }
        out << '\n';
    }
}

template <typename Robot>
void run_export(const json& problems, pRRTC_settings& settings,
                const std::string& out_dir, const std::string& robot_name) {
    using Configuration = typename Robot::Configuration;
    std::filesystem::create_directories(out_dir);

    int solved = 0, failed = 0, total = 0;
    for (const auto& data : problems) {
        if (not data["valid"]) continue;
        total++;

        std::string name = data["name"];
        auto env = problem_dict_to_env(data, name);
        Configuration start = data["start"];
        std::vector<Configuration> goals = data["goals"];

        PlannerResult<Robot> result;
        if (robot_name == "panda_four") result = pRRTC::solve<Robot>(start, goals, env, settings, 4, 8);
        else if (robot_name == "panda_dual") result = pRRTC::solve<Robot>(start, goals, env, settings, 2, 2);
        else if (robot_name == "panda_five") result = pRRTC::solve<Robot>(start, goals, env, settings, 5, 8);

        if (not result.solved) {
            failed++;
            std::cout << "[export] FAILED " << name << " (no seed written)\n";
            continue;
        }
        solved++;

        // Final (best / shortest) path -> <name>.txt : this is the seed
        // seed_from_path.py --path_dir picks up by default.
        write_path_file<Robot>(out_dir + "/" + name + ".txt", result.path);

        // Bonus: every improving solution along the way, one file each. Ordered
        // by discovery (k00 = initial, cost decreasing). Only non-empty when the
        // planner ran with rrtc_iter > 1. The downstream orchestrator sorts these
        // by length and prefers the shorter ones.
        int written = 0;
        for (std::size_t k = 0; k < result.seed_paths.size(); k++) {
            float cost = result.seed_paths[k].first;
            const auto& p = result.seed_paths[k].second;
            std::ostringstream fn;
            fn << out_dir << "/" << name << "__k" << std::setfill('0') << std::setw(2) << k
               << "__cost" << std::fixed << std::setprecision(4) << cost << ".txt";
            write_path_file<Robot>(fn.str(), p);
            written++;
        }
        std::cout << "[export] " << name << " -> " << result.path.size()
                  << " waypoints, cost " << result.cost << ", " << written
                  << " intermediate seed(s)\n";
    }
    std::cout << "[export] wrote seeds for " << solved << "/" << total
              << " problems (" << failed << " failed) into " << out_dir << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: export_seeds <robot_name> <out_dir> [rrtc_iter] [optimize_iters]\n";
        return -1;
    }
    std::string robot_name = argv[1];
    std::string out_dir = argv[2];

    pRRTC_settings settings;
    // Match evaluate_mr.cu's tuned four-arm settings...
    settings.num_new_configs = 300;
    settings.granularity = 64;
    settings.range = 2.0;
    settings.balance = 2;
    settings.tree_ratio = 1.0;
    settings.dynamic_domain = false;
    settings.dd_radius = 4.0;
    settings.dd_min_radius = 1.0;
    settings.dd_alpha = 0.0001;
    // ...but run as an OPTIMIZING planner so we get several improving paths.
    // rrtc_iter=1 -> single (initial) solution; >1 -> initial + refinements.
    settings.rrtc_iter = (argc >= 4) ? std::stoi(argv[3]) : 6;
    if (argc >= 5) settings.optimize_iters = std::stoi(argv[4]);
    settings.phs = true;           // informed sampling drives the cost down
    settings.path_simplify = true; // export the smoothed/shortcutted path

    std::string path = "scripts/" + robot_name + "_problems.json";
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Could not open " << path << " (run from the repo root)\n";
        return 1;
    }
    json problems = json::parse(f);

    if (robot_name == "panda_dual") {
        run_export<robots::Panda_dual>(problems, settings, out_dir, robot_name);
    } else if (robot_name == "panda_four") {
        run_export<robots::Panda_four>(problems, settings, out_dir, robot_name);
    } else if (robot_name == "panda_five") {
        run_export<robots::Panda_five>(problems, settings, out_dir, robot_name);
    } else {
        std::cerr << "Unsupported robot type: " << robot_name << "\n";
        return 1;
    }
    return 0;
}
