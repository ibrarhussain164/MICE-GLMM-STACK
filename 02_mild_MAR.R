# =============================================================================
# Sensitivity analysis: milder MAR mechanism
#
# Same datasets, splits and seeds as the main MAR run; only the odds-ratio
# cap of the MAR mechanism is lowered from 1:20 to 1:5 (MAR_ODDS_CAP in
# 01_main_benchmark.R). The ablation meta-learners are not fitted.
# Run from the repository root.
# =============================================================================
MISS_MECHS   <- "MAR5"
mech_labels  <- c(MAR5 = "MAR -- Missing At Random (mild, odds ratio up to 1:5)")
MISS_RATES   <- c(0.10, 0.20, 0.30, 0.40, 0.50)
RUN_ABLATION <- FALSE
OUTPUT_DIR   <- "results/mild_MAR"

source("R/01_main_benchmark.R")
