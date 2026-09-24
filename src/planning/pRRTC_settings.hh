#pragma once

struct pRRTC_settings {
    int max_samples = 1000000;
    int max_iters = 1000;
    int optimize_iters = 400000;
    int num_new_configs = 256;
    int granularity = 16;
    float range = 0.5;

    int balance = 1; // 0 = no balance, 1 = balance strategy 1, 2 = balance strategy 2
    float tree_ratio = 0.5; // 0.5 for balance=1, 1.0 for balance=2

    bool dynamic_domain = true;
    float dd_alpha = 0.0001;
    float dd_radius = 4.0;
    float dd_min_radius = 1.0;

    bool roadmap_reassign = false;

    int rrtc_iter = 1;
    int resample_iter = 5;
    bool cost_bound_resample = false;
    bool path_simplify = true;
    bool phs = true;

    int simplify_round=1;
    int bspline_step = 3;
    float bspline_min_change = 0.01;
    int perturb_step = 3;
    int perturb_empty_step = 2;

};