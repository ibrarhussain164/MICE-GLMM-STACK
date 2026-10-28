# =============================================================================
# Runtime at larger sample sizes
#
# One benchmark dataset (Pima Indians Diabetes, p = 8) is enlarged by
# stratified bootstrap resampling to n_train = 5000 and 10000, and the full
# pipeline of 01_main_benchmark.R is run at MCAR 30% (three replicates per
# size). n_train = 538, the original training size, is included as a check:
# its runtimes should be close to those of the main run. Only runtimes are of
# interest here; AUC and Brier score are not meaningful on bootstrapped data
# with duplicated rows.
#
# Run from the repository root after 01_main_benchmark.R has been checked
# (SMOKE <- TRUE first). Results are appended after every split, so an
# interrupted run can be restarted.
# =============================================================================

# --- 1. Settings ----------------------------------------------------------------
MAIN_SCRIPT   <- "R/01_main_benchmark.R"
BOOT_DATASET  <- "Pima"
N_TRAIN_GRID  <- c(538, 5000, 10000)   # 538 = original n_train of Pima
N_REPS        <- 3
BOOT_MECH     <- "MCAR"
BOOT_RATE     <- 0.30
SMOKE         <- FALSE                 # TRUE: n_train = 538 and one replicate only
OUT_DIR_BOOT  <- "results/runtime_bootstrap"

# --- 2. Load the functions of the main script (without its main loop) ----------
RUN_ABLATION <- FALSE                  # ablation fits are not part of the runtime
.lines <- readLines(MAIN_SCRIPT)
.cut   <- grep("^# --- [0-9]+\\. Main Experiment Loop", .lines)
if (length(.cut) != 1) stop("'Main Experiment Loop' section not found in ", MAIN_SCRIPT)
eval(parse(text = .lines[seq_len(.cut - 1)]), envir = globalenv())
rm(.lines, .cut)

SAVE_PREDICTIONS <- FALSE
if (SMOKE) { N_TRAIN_GRID <- c(538); N_REPS <- 1 }
if (SMOKE) OUT_DIR_BOOT <- paste0(OUT_DIR_BOOT, "-SMOKE")
dir.create(OUT_DIR_BOOT, showWarnings = FALSE, recursive = TRUE)
RAW_FILE <- file.path(OUT_DIR_BOOT, "Runtime_Bootstrap_Raw.csv")

# --- 3. Base dataset and stratified bootstrap ---------------------------------
ds_entry <- Filter(function(d) d$name == BOOT_DATASET, DATASET_REGISTRY)[[1]]
BASE <- ds_entry$loader()
PRED_NAMES <- setdiff(names(BASE), "y")
cat(sprintf("Base dataset %s: n = %d, p = %d\n", BOOT_DATASET, nrow(BASE), length(PRED_NAMES)))

# Sample with replacement within each class, so that the class balance of
# the original data is kept.
boot_dataset <- function(d, n_total, seed) {
  set.seed(seed)
  idx <- unlist(lapply(split(seq_len(nrow(d)), d$y), function(ix) {
    k <- round(n_total * length(ix) / nrow(d))
    ix[sample.int(length(ix), k, replace = TRUE)]
  }))
  out <- d[idx, , drop = FALSE]
  rownames(out) <- NULL
  out
}

# splits already finished in an earlier run are skipped
done_keys <- if (file.exists(RAW_FILE)) {
  old <- read.csv(RAW_FILE, stringsAsFactors = FALSE)
  unique(paste(old$n_train_target, old$Rep))
} else character(0)

# --- 4. Runs: each size x replicate --------------------------------------------
cat(sprintf("\n=== Runtime bootstrap: %s, %s %g%%, n_train = %s, reps = %d ===\n",
            BOOT_DATASET, BOOT_MECH, BOOT_RATE * 100,
            paste(N_TRAIN_GRID, collapse = "/"), N_REPS))
T_START <- Sys.time()
for (n_tr in N_TRAIN_GRID) {
  n_total <- ceiling(n_tr / TRAIN_FRAC)          # n_train ~ n_tr after the 70/30 split
  for (rp in seq_len(N_REPS)) {
    if (paste(n_tr, rp) %in% done_keys) {
      cat(sprintf("n_train=%d rep=%d: already done\n", n_tr, rp)); next
    }
    seed <- 5000L + n_tr + rp
    D <- boot_dataset(BASE, n_total, seed)
    set.seed(seed)
    tr_idx <- caret::createDataPartition(D$y, p = TRAIN_FRAC, list = FALSE)[, 1]
    te_idx <- setdiff(seq_len(nrow(D)), tr_idx)
    X_tr <- D[tr_idx, PRED_NAMES, drop = FALSE]; y_tr <- D[tr_idx, "y"]
    X_te <- D[te_idx, PRED_NAMES, drop = FALSE]; y_te <- D[te_idx, "y"]

    cat(sprintf("n_train=%5d rep=%d ... ", nrow(X_tr), rp)); flush.console()
    invisible(gc(reset = TRUE))
    t0 <- Sys.time()
    res <- tryCatch(run_one(BOOT_MECH, BOOT_RATE, seed, X_tr, y_tr, X_te, y_te),
                    error = function(e) { message("  [ERR] ", conditionMessage(e)); NULL })
    split_sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    if (is.null(res)) next

    tm <- res$timing
    # total for MICE-GLMM-Stack = imputation and scoring + GLMM fit
    tm <- rbind(tm,
                data.frame(Method = "MICE_GLMM_Stack_Total",
                           RunTime_sec = sum(tm$RunTime_sec[tm$Method %in%
                             c("MICE_Stack_ImputeScore", "MICE_GLMM_Stack_MetaFit")])),
                data.frame(Method = "Split_Total_noAblation", RunTime_sec = split_sec))
    tm$Dataset <- BOOT_DATASET; tm$MissMech <- BOOT_MECH; tm$MissRate <- BOOT_RATE
    tm$n_train_target <- n_tr; tm$n_train <- nrow(X_tr); tm$p <- length(PRED_NAMES)
    tm$Rep <- rp
    tm$Peak_memory_MB <- round(sum(gc()[, ncol(gc())]), 1)
    tm$Finished_at <- format(Sys.time(), "%Y-%m-%d %H:%M")
    write.table(tm, RAW_FILE, sep = ",", row.names = FALSE,
                col.names = !file.exists(RAW_FILE), append = file.exists(RAW_FILE))
    cat(sprintf("%.1f s (Stack total %.1f s)\n", split_sec,
                tm$RunTime_sec[tm$Method == "MICE_GLMM_Stack_Total"]))
  }
}
cat(sprintf("\n=== Finished: %.1f min ===\n",
            as.numeric(difftime(Sys.time(), T_START, units = "mins"))))

# --- 5. Summary -------------------------------------------------------------------
raw <- read.csv(RAW_FILE, stringsAsFactors = FALSE)
summ <- raw %>%
  group_by(Method, n_train_target) %>%
  summarise(N = n(), n_train = round(mean(n_train)), p = first(p),
            Mean_sec = mean(RunTime_sec), SD_sec = sd(RunTime_sec),
            Min_sec = min(RunTime_sec), Max_sec = max(RunTime_sec),
            .groups = "drop")

# Check at n_train = 538: mean runtime of the 10 Pima splits at MCAR 30% in
# the main run (Multi_Runtime_Raw.csv). Split_Total_noAblation is the split
# total minus the ablation fits.
REAL_PIMA_538 <- c(CCA = 0.077, MeanSI = 0.175, KNN = 0.440, RFI = 0.655,
                   `MIE-SE` = 7.175, MICE_Stack_ImputeScore = 31.595,
                   MICE_GLMM_Stack_MetaFit = 2.530, MICE_GLMM_Stack_Total = 34.125,
                   Split_Total_noAblation = 42.660)
summ$Real_Pima538_sec <- ifelse(summ$n_train_target == 538 & BOOT_DATASET == "Pima",
                                unname(REAL_PIMA_538[summ$Method]), NA_real_)
summ$Boot_over_real <- round(summ$Mean_sec / summ$Real_Pima538_sec, 2)

# scaling exponent in n (p is fixed): slope of log(time) on log(n_train)
expo <- raw %>%
  group_by(Method) %>%
  summarise(Exponent_n = if (n_distinct(n_train) > 1)
              unname(coef(lm(log(RunTime_sec) ~ log(n_train)))[2]) else NA_real_,
            .groups = "drop")
summ <- left_join(summ, expo, by = "Method")

ord <- c("CCA", "MeanSI", "KNN", "RFI", "MIE-SE", "MICE_Stack_ImputeScore",
         "MICE_GLMM_Stack_MetaFit", "MICE_GLMM_Stack_Total", "Split_Total_noAblation")
summ <- summ %>% filter(Method %in% ord) %>%
  mutate(Method = factor(Method, levels = ord)) %>% arrange(Method, n_train_target)
write.csv(summ, file.path(OUT_DIR_BOOT, "Runtime_Bootstrap_Summary.csv"), row.names = FALSE)

cat("\n--- Runtime per split (sec), mean over replicates ---\n")
print(as.data.frame(summ %>% transmute(Method, n_train, N,
                                       Mean = round(Mean_sec, 1), SD = round(SD_sec, 1),
                                       Real538 = round(Real_Pima538_sec, 1),
                                       Boot_Real = Boot_over_real,
                                       Exp_n = round(Exponent_n, 2))), row.names = FALSE)
cat(sprintf("\nFiles: %s\n", OUT_DIR_BOOT))
