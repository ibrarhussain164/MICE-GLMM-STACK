# =============================================================================
# Multi_HAS_Pipeline.R
# MICE-GLMM-Stack: Pooled 3-Random-Effect GLMM Stacking Benchmark
# 15 Numeric-Predictor Binary Clinical / Epidemiological Datasets
#
# Authors  : Ibrar Hussain, Sohail Akhtar, Xinyi Cao, Can Liu,
#            Yiqing Gu, Zaixiang Tang
# Contact  : tangzx@suda.edu.cn
# Repository: https://github.com/ibrarhussain164/MICE-GLMM-STACK
#
# Datasets (15):
#   1  Pima Diabetes           n=768  p=8
#   2  WDBC Breast Cancer      n=569  p=30
#   3  Blood Transfusion       n=748  p=4
#   4  Diabetic Retinopathy    n=1151 p=16
#   5  Breast Cancer Coimbra   n=116  p=9
#   6  Vertebral Column        n=310  p=6
#   7  Haberman Survival       n=306  p=3
#   8  ILPD Liver Patient      n=583  p=9
#   9  Heart Failure           n=299  p=6
#  10  Cleveland Heart Disease n=297  p=6
#  11  Parkinson Disease       n=195  p=22
#  12  South African Heart     n=462  p=8
#  13  Statlog Heart Disease   n=270  p=6
#  14  HCV Hepatitis C         n~589  p=11
#  15  WPBC Breast Prognostic  n=194  p=30
#
# Methods compared:
#   Complete        : No missingness (reference baseline)
#   MeanSI          : Mean single imputation
#   KNN             : k-NN single imputation (VIM)
#   RFI             : Random forest single imputation (missForest)
#   CCA             : Complete case analysis
#   MICE-PMM        : MICE predictive mean matching (m=5, averaged)
#   MIE-SE          : Stacking ensemble (Aleryani et al. 2020)
#   MICE-GLMM-Stack : Proposed pooled 3-random-effect GLMM meta-learner
#
# MICE-GLMM-Stack pools: 5 classifiers x 4 MICE methods x 5 draws = 100 streams
# Random effects: (1) classifier slope, (2) method intercept, (3) draw-in-method
# Formula: y ~ logit_p + (logit_p | clf_id) + (1 | method_id/draw_id)
# =============================================================================


# =============================================================================
# 0. PACKAGES
# =============================================================================

suppressPackageStartupMessages({
  library(mice)
  library(VIM)
  library(missForest)
  library(C50)
  library(naivebayes)
  library(e1071)
  library(randomForest)
  library(rpart)
  library(lme4)
  library(pROC)
  library(caret)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(purrr)
})
set.seed(2024)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

N_SPLITS           <- 10
MISS_RATES         <- c(0.10, 0.20, 0.30, 0.40, 0.50)
M_IMP              <- 5
M_IMP_SE           <- 5
MAXIT_MICE         <- 20
CV_FOLDS           <- 10
TRAIN_FRAC         <- 0.70
CLF_NAMES          <- c("J48", "NB", "PART", "SMO", "RF")
MISS_MECHS         <- c("MCAR", "MAR")
MICE_STACK_METHODS <- c("pmm", "cart", "norm", "rf")
M_IMP_STACK        <- M_IMP
MICE_METHOD_LABEL  <- c(pmm  = "MICE-PMM",  cart = "MICE-CART",
                        norm = "MICE-Norm", rf   = "MICE-RF")
METHOD_ORDER       <- c("Complete", "MeanSI", "KNN", "RFI", "CCA",
                        "MICE-PMM", "MICE-GLMM-Stack", "MIE-SE")

mech_labels <- c(MCAR = "MCAR — Completely At Random",
                 MAR  = "MAR — Missing At Random (severity-weighted)")

OUTPUT_DIR <- "~/Desktop/project4/GLMM-stack/GLMM-stack/GLMM-stack-mice"
DATA_DIR   <- "~/Desktop/project4/GLMM-stack/data"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)


# =============================================================================
# 2. DATASET LOADING
# =============================================================================

read_ds <- function(name) {
  d <- read.csv(file.path(DATA_DIR, paste0(name, ".csv")),
                stringsAsFactors = FALSE)

  # Locate outcome column
  if (!"y" %in% names(d)) {
    candidates <- c("y", "Y", "outcome", "Outcome", "OUTCOME", "target",
                    "Target", "TARGET", "class", "Class", "CLASS",
                    "label", "Label", "LABEL", "DEATH_EVENT", "death_event",
                    "Death_Event", "diagnosis", "Diagnosis", "DIAGNOSIS",
                    "status", "Status", "STATUS", "event", "Event", "EVENT")
    found <- intersect(candidates, names(d))
    if (length(found) == 0)
      stop(sprintf("[%s] no outcome column found (columns: %s).",
                   name, paste(names(d), collapse = ", ")))
    names(d)[names(d) == found[1]] <- "y"
    cat(sprintf("[%s] outcome column '%s' renamed to 'y'\n", name, found[1]))
  }

  # Drop survival-time columns
  time_cols <- intersect(c("time", "Time", "TIME", "followup_time",
                           "follow_up_time", "FollowUpTime"), names(d))
  if (length(time_cols) > 0) {
    d <- d[, !(names(d) %in% time_cols), drop = FALSE]
    cat(sprintf("[%s] dropped: %s\n", name, paste(time_cols, collapse = ", ")))
  }

  # Coerce outcome to neg/pos
  y_chr_raw <- toupper(trimws(as.character(d$y)))
  y_chr     <- y_chr_raw
  if (!all(y_chr %in% c("NEG", "POS", "NA", ""))) {
    y_chr <- dplyr::case_when(
      y_chr %in% c("1", "Y", "T", "R", "YES", "TRUE", "POS", "POSITIVE",
                   "EVENT", "DEAD", "DEATH", "PRESENT", "RECURRENCE")    ~ "pos",
      y_chr %in% c("0", "N", "F", "NO", "FALSE", "NEG", "NEGATIVE",
                   "ALIVE", "SURVIVED", "ABSENT", "NONRECURRENCE",
                   "NON-RECURRENCE")                                     ~ "neg",
      TRUE ~ NA_character_)
    d$y <- y_chr
  }
  d$y <- factor(tolower(trimws(as.character(d$y))), levels = c("neg", "pos"))

  unmapped <- setdiff(unique(y_chr_raw[is.na(d$y)]), c("NA", ""))
  if (length(unmapped) > 0)
    warning(sprintf("[%s] unmapped outcome values: %s",
                    name, paste(unmapped, collapse = ", ")), call. = FALSE)

  # Drop rows with pre-existing missingness
  n0 <- nrow(d)
  d  <- d[stats::complete.cases(d), , drop = FALSE]
  if (nrow(d) < n0)
    cat(sprintf("[%s] dropped %d rows with original NAs\n", name, n0 - nrow(d)))

  # Class balance check
  tab <- table(d$y)
  if (length(tab) < 2 || any(tab < 2))
    stop(sprintf("[%s] not usably binary after coercion (counts: %s)",
                 name, paste(names(tab), tab, sep = "=", collapse = ", ")))
  d
}

load_pima          <- function() read_ds("Pima")
load_wdbc          <- function() read_ds("WDBC_Breast")
load_transfusion   <- function() read_ds("BloodTransfusion")
load_retinopathy   <- function() read_ds("DiabeticRetinopathy")
load_coimbra       <- function() read_ds("CoimbraBreast")
load_vertebral     <- function() read_ds("Vertebral")
load_haberman      <- function() read_ds("Haberman")
load_ilpd          <- function() read_ds("ILPD_Liver")
load_heart_failure <- function() read_ds("HeartFailure")
load_cleveland     <- function() read_ds("ClevelandHeart")
load_parkinsons    <- function() read_ds("Parkinsons")
load_saheart       <- function() read_ds("SAHeart")
load_statlogheart  <- function() read_ds("StatlogHeart")
load_hcvhepc       <- function() read_ds("HCVHepC")
load_wpbc          <- function() read_ds("WPBCBreast")

DATASET_REGISTRY <- list(
  list(name = "Pima",                loader = load_pima),
  list(name = "WDBC_Breast",         loader = load_wdbc),
  list(name = "BloodTransfusion",    loader = load_transfusion),
  list(name = "DiabeticRetinopathy", loader = load_retinopathy),
  list(name = "CoimbraBreast",       loader = load_coimbra),
  list(name = "Vertebral",           loader = load_vertebral),
  list(name = "Haberman",            loader = load_haberman),
  list(name = "ILPD_Liver",          loader = load_ilpd),
  list(name = "HeartFailure",        loader = load_heart_failure),
  list(name = "ClevelandHeart",      loader = load_cleveland),
  list(name = "Parkinsons",          loader = load_parkinsons),
  list(name = "SAHeart",             loader = load_saheart),
  list(name = "StatlogHeart",        loader = load_statlogheart),
  list(name = "HCVHepC",             loader = load_hcvhepc),
  list(name = "WPBCBreast",          loader = load_wpbc)
)


# =============================================================================
# 3. MISSINGNESS INDUCTION
# =============================================================================

induce_missing <- function(X, rate, mech, seed = NULL) {
  if (rate == 0) return(X)
  mech_up <- toupper(mech)

  if (!is.null(seed)) {
    mech_offset <- switch(mech_up, MCAR = 1009L, MAR = 7919L, 4001L)
    set.seed(as.integer(seed) + mech_offset)
  }

  p           <- ncol(X)
  min_observed <- if (p >= 4L) 2L else 1L
  k_needed    <- max(1L, min(p - min_observed, ceiling(rate * p * 1.3) + 1L))
  patterns_mat <- matrix(1L, nrow = p, ncol = p)
  for (i in seq_len(p)) {
    miss_cols <- ((i - 1L) + 0L:(k_needed - 1L)) %% p + 1L
    patterns_mat[i, miss_cols] <- 0L
  }

  amp_err    <- NULL
  amp_result <- suppressMessages(suppressWarnings(tryCatch({
    if (mech_up == "MAR") {
      case_frac <- min(0.999, rate * p / k_needed)
      if (case_frac <= 0.5) {
        odds_ratio <- 20
      } else {
        x_sat      <- 0.5 / case_frac
        odds_ratio <- max(1.01, min(20, x_sat / (1 - x_sat)))
      }
      strong_odds <- matrix(rep(c(1, odds_ratio), p), nrow = p, byrow = TRUE)
      amp <- mice::ampute(X, prop = rate, mech = "MAR", bycases = FALSE,
                          patterns = patterns_mat, cont = FALSE,
                          odds = strong_odds)
    } else {
      amp <- mice::ampute(X, prop = rate, mech = mech_up, bycases = FALSE,
                          patterns = patterns_mat)
    }
    amp$amp
  }, error = function(e) { amp_err <<- conditionMessage(e); NULL })))

  if (!is.null(amp_result)) return(amp_result)

  # Fallback: manual cell-level missingness
  message(sprintf("  [induce_missing] ampute() failed (%s) — using fallback",
                  if (is.null(amp_err)) "unknown" else amp_err))
  n_cells_total <- nrow(X) * ncol(X)
  n_miss_total  <- max(1L, floor(rate * n_cells_total))
  X_miss        <- X

  if (mech_up == "MAR") {
    z        <- scale(as.matrix(X)); z[!is.finite(z)] <- 0
    severity <- rowMeans(z)
    rank_pct <- rank(severity, ties.method = "average") / length(severity)
    odds_row <- 1 + 19 * rank_pct
    p_row    <- odds_row / sum(odds_row)
    cell_prob <- rep(p_row, ncol(X))
    cell_idx  <- sample.int(n_cells_total, size = n_miss_total,
                            prob = cell_prob, replace = FALSE)
  } else {
    cell_idx <- sample.int(n_cells_total, size = n_miss_total, replace = FALSE)
  }

  row_i <- ((cell_idx - 1L) %% nrow(X)) + 1L
  col_j <- ((cell_idx - 1L) %/% nrow(X)) + 1L
  for (k in seq_along(cell_idx)) X_miss[row_i[k], col_j[k]] <- NA
  X_miss
}


# =============================================================================
# 4. IMPUTATION METHODS
# =============================================================================

impute_mean <- function(X_tr_miss, X_te_full = NULL) {
  fill_vals <- sapply(X_tr_miss, function(col) {
    fv <- mean(col, na.rm = TRUE); if (is.na(fv)) 0 else fv
  })
  fill_df <- function(df) {
    for (nm in names(df)) {
      idx <- is.na(df[[nm]]); if (!any(idx)) next
      df[[nm]][idx] <- fill_vals[nm]
    }
    df
  }
  list(train = fill_df(X_tr_miss),
       test  = if (is.null(X_te_full)) NULL else X_te_full)
}

impute_knn <- function(X_tr_miss, X_te_full = NULL, k = 5) {
  suppressMessages(suppressWarnings({
    list(train = VIM::kNN(X_tr_miss, k = k, imp_var = FALSE),
         test  = if (is.null(X_te_full)) NULL else X_te_full)
  }))
}

impute_rfi <- function(X_tr_miss, X_te_full = NULL) {
  suppressMessages(suppressWarnings({
    list(train = missForest::missForest(X_tr_miss, verbose = FALSE)$ximp,
         test  = if (is.null(X_te_full)) NULL else X_te_full)
  }))
}

mice_impute <- function(X_miss, m = M_IMP, y_vec = NULL,
                        seed = 1, maxit = MAXIT_MICE) {
  df    <- if (!is.null(y_vec)) data.frame(X_miss, y = y_vec) else X_miss
  meths <- setNames(rep("pmm", ncol(df)), names(df))
  meths["y"] <- ""
  for (nm in names(df)) if (!any(is.na(df[[nm]]))) meths[nm] <- ""
  suppressMessages(suppressWarnings(
    mids <- mice::mice(df, m = m, method = meths,
                       seed = seed, printFlag = FALSE, maxit = maxit)
  ))
  lapply(seq_len(m), function(i) {
    comp <- mice::complete(mids, i)
    comp[, setdiff(names(comp), "y"), drop = FALSE]
  })
}

mice_impute_method <- function(X_miss, method_name, m = M_IMP_STACK,
                               y_vec = NULL, seed = 1, maxit = MAXIT_MICE) {
  df    <- if (!is.null(y_vec)) data.frame(X_miss, y = y_vec) else X_miss
  meths <- setNames(rep(method_name, ncol(df)), names(df))
  meths["y"] <- ""
  for (nm in names(df)) if (!any(is.na(df[[nm]]))) meths[nm] <- ""
  suppressMessages(suppressWarnings(
    mids <- mice::mice(df, m = m, method = meths,
                       seed = seed, printFlag = FALSE, maxit = maxit)
  ))
  lapply(seq_len(m), function(i) {
    comp <- mice::complete(mids, i)
    comp[, setdiff(names(comp), "y"), drop = FALSE]
  })
}


# =============================================================================
# 5. PERFORMANCE METRICS
# =============================================================================

compute_metrics <- function(probs, labels, thresh = 0.5) {
  probs  <- pmin(pmax(as.numeric(probs), 0), 1)
  lb     <- as.integer(labels == "pos")
  auc    <- tryCatch(as.numeric(pROC::auc(pROC::roc(lb, probs, quiet = TRUE))),
                     error = function(e) NA_real_)
  br     <- mean((probs - lb)^2, na.rm = TRUE)
  pred_p <- (probs >= thresh); act_p <- (lb == 1)
  tp <- sum( pred_p &  act_p); fp <- sum( pred_p & !act_p)
  tn <- sum(!pred_p & !act_p); fn <- sum(!pred_p &  act_p)
  se <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  sp <- if ((tn + fp) == 0) NA_real_ else tn / (tn + fp)
  c(AUC = auc, Brier = br, Sensitivity = se, Specificity = sp,
    Threshold = thresh)
}


# =============================================================================
# 6. BASE CLASSIFIERS
# =============================================================================

fit_clf <- function(clf, X_tr, y_tr, X_te) {
  df_tr <- data.frame(X_tr, y = y_tr)
  df_te <- data.frame(X_te)

  out <- suppressWarnings(tryCatch({
    if (clf == "J48") {
      m <- C5.0(y ~ ., df_tr, rules = FALSE)
      predict(m, df_te, type = "prob")[, "pos"]
    } else if (clf == "NB") {
      m <- naivebayes::naive_bayes(y ~ ., df_tr, laplace = 1)
      predict(m, df_te, type = "prob")[, "pos"]
    } else if (clf == "PART") {
      m <- C5.0(y ~ ., df_tr, rules = TRUE)
      predict(m, df_te, type = "prob")[, "pos"]
    } else if (clf == "SMO") {
      x_nms <- setdiff(names(df_tr), "y")
      keep  <- x_nms[vapply(df_tr[x_nms], function(v) {
        sv <- sd(v, na.rm = TRUE); is.finite(sv) && sv > 1e-8
      }, logical(1))]
      if (length(keep) == 0) stop("SMO: no valid predictors")
      m <- e1071::svm(df_tr[, keep, drop = FALSE], df_tr$y,
                      kernel = "linear", probability = TRUE)
      attr(predict(m, df_te[, keep, drop = FALSE], probability = TRUE),
           "probabilities")[, "pos"]
    } else if (clf == "RF") {
      m <- randomForest(y ~ ., df_tr, ntree = 200)
      predict(m, df_te, type = "prob")[, "pos"]
    } else {
      rep(0.5, nrow(df_te))
    }
  }, error = function(e) rep(0.5, nrow(df_te))))

  n_te <- nrow(df_te)
  if (is.null(out) || length(out) != n_te) return(rep(0.5, n_te))
  out <- as.numeric(out)
  out[!is.finite(out)] <- 0.5
  pmin(pmax(out, 0), 1)
}


# =============================================================================
# 7. MIE-SE (ALERYANI ET AL. 2020)
# =============================================================================

mice_se_l0 <- function(imp_tr, y_tr, imp_te,
                       m_se = M_IMP_SE, oof_folds = CV_FOLDS) {
  m_use   <- min(m_se, length(imp_tr))
  n_tr    <- length(y_tr)
  n_te    <- nrow(imp_te[[1]])
  col_nms <- as.vector(outer(CLF_NAMES, seq_len(m_use),
                             FUN = function(c, k) paste0(c, "_m", k)))

  L0_tr <- matrix(0.5, n_tr, length(col_nms), dimnames = list(NULL, col_nms))
  k_use <- min(oof_folds, n_tr)
  folds <- caret::createFolds(y_tr, k = k_use, list = TRUE, returnTrain = FALSE)

  for (cl in CLF_NAMES)
    for (k in seq_len(m_use)) {
      nm <- paste0(cl, "_m", k)
      for (fi in folds) {
        tri <- setdiff(seq_len(n_tr), fi)
        L0_tr[fi, nm] <- tryCatch(
          fit_clf(cl, imp_tr[[k]][tri, , drop = FALSE],
                  y_tr[tri],
                  imp_tr[[k]][fi,  , drop = FALSE]),
          error = function(e) rep(0.5, length(fi)))
      }
    }

  L0_te <- matrix(0.5, n_te, length(col_nms), dimnames = list(NULL, col_nms))
  for (cl in CLF_NAMES)
    for (k in seq_len(m_use)) {
      nm <- paste0(cl, "_m", k)
      L0_te[, nm] <- tryCatch(
        fit_clf(cl, imp_tr[[k]], y_tr, imp_te[[k]]),
        error = function(e) rep(0.5, n_te))
    }

  L0_tr[is.na(L0_tr)] <- 0.5
  L0_te[is.na(L0_te)] <- 0.5
  list(L0_tr = L0_tr, L0_te = L0_te)
}

run_mice_se <- function(L0_tr, L0_te, y_tr, meta_clf) {
  df_tr <- as.data.frame(L0_tr)
  df_te <- as.data.frame(L0_te)
  tryCatch(
    fit_clf(meta_clf, df_tr, y_tr, df_te),
    error = function(e) rowMeans(L0_te, na.rm = TRUE))
}


# =============================================================================
# 8. MICE BASE-METHOD IMPUTATIONS
# =============================================================================

mice_stack_imputations <- function(X_tr_miss, y_tr, split_seed) {
  setNames(lapply(seq_along(MICE_STACK_METHODS), function(mi) {
    meth <- MICE_STACK_METHODS[mi]
    tryCatch(
      mice_impute_method(X_tr_miss, meth, M_IMP_STACK, y_tr, split_seed + mi),
      error = function(e) NULL)
  }), MICE_STACK_METHODS)
}

mice_stack_method_scores <- function(clf, imp_draws_list, y_tr, X_te_full,
                                     cv_folds = CV_FOLDS) {
  n_tr  <- length(y_tr)
  meths <- MICE_STACK_METHODS
  oof_mat   <- matrix(0.5, n_tr,            length(meths),
                      dimnames = list(NULL, meths))
  test_mat  <- matrix(0.5, nrow(X_te_full), length(meths),
                      dimnames = list(NULL, meths))
  oof_draws  <- setNames(vector("list", length(meths)), meths)
  test_draws <- setNames(vector("list", length(meths)), meths)

  folds <- caret::createFolds(y_tr, k = cv_folds, list = TRUE,
                              returnTrain = FALSE)

  for (mi in seq_along(meths)) {
    meth      <- meths[mi]
    imp_draws <- imp_draws_list[[meth]]
    if (is.null(imp_draws)) next

    oof_draw_mat <- matrix(NA_real_, n_tr, length(imp_draws))
    for (di in seq_along(imp_draws)) {
      draw <- imp_draws[[di]]
      for (fi in folds) {
        tri <- setdiff(seq_len(n_tr), fi)
        oof_draw_mat[fi, di] <- tryCatch(
          fit_clf(clf, draw[tri, , drop = FALSE],
                  y_tr[tri],
                  draw[fi,  , drop = FALSE]),
          error = function(e) rep(0.5, length(fi)))
      }
    }
    oof_mat[, mi]     <- rowMeans(oof_draw_mat, na.rm = TRUE)
    oof_draws[[meth]] <- oof_draw_mat

    test_draw_mat <- matrix(NA_real_, nrow(X_te_full), length(imp_draws))
    for (di in seq_along(imp_draws))
      test_draw_mat[, di] <- tryCatch(
        fit_clf(clf, imp_draws[[di]], y_tr, X_te_full),
        error = function(e) rep(0.5, nrow(X_te_full)))

    test_mat[, mi]      <- rowMeans(test_draw_mat, na.rm = TRUE)
    test_draws[[meth]]  <- test_draw_mat
  }

  oof_mat[is.na(oof_mat)]   <- 0.5
  test_mat[is.na(test_mat)] <- 0.5
  list(oof = oof_mat, test = test_mat,
       oof_draws = oof_draws, test_draws = test_draws)
}


# =============================================================================
# 9. MICE-GLMM-STACK (PROPOSED METHOD)
# =============================================================================

build_pooled_glmm_stack_data <- function(stack_scores_by_clf, y_tr) {
  rows_tr <- list()
  rows_te <- list()

  for (cl in names(stack_scores_by_clf)) {
    sc <- stack_scores_by_clf[[cl]]
    if (is.null(sc)) next
    for (meth in MICE_STACK_METHODS) {
      oof_mat  <- sc$oof_draws[[meth]]
      test_mat <- sc$test_draws[[meth]]
      if (is.null(oof_mat) || is.null(test_mat)) next
      for (di in seq_len(ncol(oof_mat))) {
        rows_tr[[length(rows_tr) + 1]] <- data.frame(
          p         = oof_mat[, di],
          y_bin     = as.integer(y_tr == "pos"),
          draw_id   = factor(di,   levels = seq_len(M_IMP_STACK)),
          method_id = factor(meth, levels = MICE_STACK_METHODS),
          clf_id    = factor(cl,   levels = CLF_NAMES),
          stringsAsFactors = FALSE)
        rows_te[[length(rows_te) + 1]] <- data.frame(
          p         = test_mat[, di],
          draw_id   = factor(di,   levels = seq_len(M_IMP_STACK)),
          method_id = factor(meth, levels = MICE_STACK_METHODS),
          clf_id    = factor(cl,   levels = CLF_NAMES),
          obs_id    = seq_len(nrow(test_mat)),
          stringsAsFactors = FALSE)
      }
    }
  }

  if (length(rows_tr) == 0 || length(rows_te) == 0) return(NULL)

  long_tr <- do.call(rbind, rows_tr)
  long_tr$logit_p <- qlogis(pmin(pmax(long_tr$p, 1e-6), 1 - 1e-6))
  long_te <- do.call(rbind, rows_te)
  long_te$logit_p <- qlogis(pmin(pmax(long_te$p, 1e-6), 1 - 1e-6))

  list(long_tr = long_tr, long_te = long_te)
}

run_mice_glmm_stack_pooled <- function(long_data, test_list_fallback) {
  if (is.null(long_data))
    return(list(
      pred         = rowMeans(do.call(cbind, test_list_fallback), na.rm = TRUE),
      varcomp      = NULL,
      formula_used = 4L))

  long_tr <- long_data$long_tr
  long_te <- long_data$long_te

  mk_ctrl <- function(maxfun)
    glmerControl(optimizer = "bobyqa",
                 optCtrl   = list(maxfun = maxfun),
                 check.conv.singular = .makeCC("ignore", tol = 1e-3))

  # Fallback ladder:
  #   Rung 1: Random slope + intercept by classifier; draw nested in method
  #   Rung 2: Random intercept by classifier; draw nested in method
  #   Rung 3: Crossed intercepts (classifier, method, draw)
  #   Rung 4: Row-mean (unweighted average)
  formula_ladder <- list(
    list(f    = y_bin ~ logit_p + (logit_p | clf_id) + (1 | method_id/draw_id),
         ctrl = mk_ctrl(5e5)),
    list(f    = y_bin ~ logit_p + (1 | clf_id) + (1 | method_id/draw_id),
         ctrl = mk_ctrl(4e5)),
    list(f    = y_bin ~ logit_p + (1 | clf_id) + (1 | method_id) + (1 | draw_id),
         ctrl = mk_ctrl(3e5))
  )

  ok <- FALSE; mod <- NULL; formula_used <- NA_integer_
  for (fi in seq_along(formula_ladder)) {
    if (ok) break
    fc <- formula_ladder[[fi]]
    suppressMessages(suppressWarnings(tryCatch({
      mod          <- lme4::glmer(fc$f, data = long_tr,
                                  family = binomial(), control = fc$ctrl)
      ok           <- TRUE
      formula_used <- fi
    }, error = function(e) NULL)))
  }

  if (!ok) {
    agg_fb <- aggregate(p ~ obs_id, long_te[, c("obs_id", "p")], mean)
    return(list(pred         = agg_fb[order(agg_fb$obs_id), "p"],
                varcomp      = NULL,
                formula_used = 4L))
  }

  long_te$pred <- tryCatch(
    predict(mod, long_te, type = "response",
            allow.new.levels = TRUE, re.form = NULL),
    error = function(e)
      plogis(predict(mod, long_te, type = "link",
                     allow.new.levels = TRUE, re.form = NULL)))

  agg <- aggregate(pred ~ obs_id, long_te, mean)

  vc_df <- tryCatch({
    vc_raw <- as.data.frame(lme4::VarCorr(mod))
    vc_raw[, c("grp", "var1", "var2", "vcov", "sdcor")]
  }, error = function(e) NULL)

  list(pred         = agg[order(agg$obs_id), "pred"],
       varcomp      = vc_df,
       formula_used = formula_used)
}


# =============================================================================
# 10. SINGLE RUN
# =============================================================================

run_one <- function(miss_mech, miss_rate, split_seed,
                    X_tr_full, y_tr, X_te_full, y_te) {
  rows         <- list()
  varcomp_local <- list()
  timing_local  <- list()

  rec <- function(method, clf, test_probs) {
    mt <- compute_metrics(test_probs, y_te)
    data.frame(Method = method, Classifier = clf, t(mt),
               stringsAsFactors = FALSE)
  }
  add_timing <- function(label, secs)
    timing_local[[length(timing_local) + 1]] <<- data.frame(
      Method = label, RunTime_sec = as.numeric(secs),
      stringsAsFactors = FALSE)

  X_tr_miss <- tryCatch(
    induce_missing(X_tr_full, miss_rate, miss_mech, seed = split_seed),
    error = function(e) X_tr_full)

  # Single imputation baselines
  si_mean_res <- impute_mean(X_tr_miss, X_te_full)
  si_knn_res  <- tryCatch(impute_knn(X_tr_miss, X_te_full),
                           error = function(e) impute_mean(X_tr_miss, X_te_full))
  si_rfi_res  <- tryCatch(impute_rfi(X_tr_miss, X_te_full),
                           error = function(e) impute_mean(X_tr_miss, X_te_full))

  for (cl in CLF_NAMES) {
    rows[[length(rows)+1]] <- rec("MeanSI", cl,
      fit_clf(cl, si_mean_res$train, y_tr, si_mean_res$test))
    rows[[length(rows)+1]] <- rec("KNN", cl,
      fit_clf(cl, si_knn_res$train,  y_tr, si_knn_res$test))
    rows[[length(rows)+1]] <- rec("RFI", cl,
      fit_clf(cl, si_rfi_res$train,  y_tr, si_rfi_res$test))
  }

  # CCA
  cca_keep <- stats::complete.cases(X_tr_miss)
  X_tr_cca <- X_tr_miss[cca_keep, , drop = FALSE]
  y_tr_cca <- y_tr[cca_keep]
  cca_ok   <- nrow(X_tr_cca) >= 10 &&
    length(unique(y_tr_cca)) == 2 &&
    all(table(y_tr_cca) >= 2)

  for (cl in CLF_NAMES) {
    p_cca <- if (cca_ok) {
      tryCatch(fit_clf(cl, X_tr_cca, y_tr_cca, X_te_full),
               error = function(e) rep(0.5, nrow(X_te_full)))
    } else rep(0.5, nrow(X_te_full))
    rows[[length(rows)+1]] <- rec("CCA", cl, p_cca)
  }

  # MIE-SE
  .t0_se    <- Sys.time()
  imp_tr_se <- mice_impute(X_tr_miss, M_IMP_SE, y_tr, split_seed + 1L, MAXIT_MICE)
  imp_te_se <- lapply(seq_len(M_IMP_SE), function(i) X_te_full)
  l0        <- tryCatch(mice_se_l0(imp_tr_se, y_tr, imp_te_se),
                        error = function(e) NULL)

  for (meta_cl in CLF_NAMES) {
    p_se <- if (!is.null(l0)) {
      tryCatch(run_mice_se(l0$L0_tr, l0$L0_te, y_tr, meta_cl),
               error = function(e) rowMeans(l0$L0_te, na.rm = TRUE))
    } else {
      rowMeans(do.call(cbind, lapply(seq_len(M_IMP_SE), function(k)
        fit_clf(meta_cl, imp_tr_se[[k]], y_tr, X_te_full))), na.rm = TRUE)
    }
    rows[[length(rows)+1]] <- rec("MIE-SE", meta_cl, p_se)
  }
  add_timing("MIE-SE",
             as.numeric(difftime(Sys.time(), .t0_se, units = "secs")))

  # MICE base-method imputations (shared across classifiers)
  .t0_imp       <- Sys.time()
  mice_stack_draws <- tryCatch(
    mice_stack_imputations(X_tr_miss, y_tr, split_seed + 100L),
    error = function(e) NULL)

  stack_scores_by_clf <- list()
  if (!is.null(mice_stack_draws)) {
    for (cl in CLF_NAMES) {
      sc <- tryCatch(
        mice_stack_method_scores(cl, mice_stack_draws, y_tr, X_te_full),
        error = function(e) NULL)

      if (!is.null(sc)) {
        stack_scores_by_clf[[cl]] <- sc
        # Report MICE-PMM as standalone competitor row (only pmm)
        for (mi in seq_along(MICE_STACK_METHODS)) {
          if (MICE_STACK_METHODS[mi] != "pmm") next
          rows[[length(rows)+1]] <- rec(
            MICE_METHOD_LABEL[["pmm"]], cl, sc$test[, mi])
        }
      } else {
        # Fallback: direct per-draw fit for MICE-PMM row
        imp_draws <- mice_stack_draws[["pmm"]]
        p_pmm <- if (!is.null(imp_draws)) {
          tryCatch(rowMeans(do.call(cbind, lapply(imp_draws, function(draw)
            fit_clf(cl, draw, y_tr, X_te_full))), na.rm = TRUE),
            error = function(e) rep(0.5, nrow(X_te_full)))
        } else rep(0.5, nrow(X_te_full))
        rows[[length(rows)+1]] <- rec(MICE_METHOD_LABEL[["pmm"]], cl, p_pmm)
      }
    }
  }
  add_timing("MICE_Stack_ImputeScore",
             as.numeric(difftime(Sys.time(), .t0_imp, units = "secs")))

  # MICE-GLMM-Stack
  if (length(stack_scores_by_clf) > 0) {
    .t0_glmm     <- Sys.time()
    fallback_list <- lapply(stack_scores_by_clf, function(sc)
      rowMeans(sc$test, na.rm = TRUE))
    glmm_data    <- tryCatch(
      build_pooled_glmm_stack_data(stack_scores_by_clf, y_tr),
      error = function(e) NULL)
    glmm_result  <- tryCatch(
      run_mice_glmm_stack_pooled(glmm_data, fallback_list),
      error = function(e) list(
        pred         = rowMeans(do.call(cbind, fallback_list), na.rm = TRUE),
        varcomp      = NULL,
        formula_used = NA_integer_))

    rows[[length(rows)+1]] <- rec("MICE-GLMM-Stack", "All", glmm_result$pred)

    if (!is.null(glmm_result$varcomp) && nrow(glmm_result$varcomp) > 0)
      varcomp_local[[length(varcomp_local) + 1]] <- data.frame(
        FormulaRung = glmm_result$formula_used,
        glmm_result$varcomp, stringsAsFactors = FALSE)

    add_timing("MICE_GLMM_Stack_MetaFit",
               as.numeric(difftime(Sys.time(), .t0_glmm, units = "secs")))
  }

  list(scores  = do.call(rbind, rows),
       timing  = do.call(rbind, timing_local),
       varcomp = do.call(rbind, varcomp_local))
}


# =============================================================================
# 11. MAIN EXPERIMENT LOOP
# =============================================================================

cat("\n=== MICE-GLMM-Stack Benchmark ===\n")
cat(sprintf("Classifiers: %s | Splits: %d | M_IMP: %d\n",
            paste(CLF_NAMES, collapse = "/"), N_SPLITS, M_IMP))

all_results <- list()
all_timings <- list()
all_varcomp <- list()

for (ds in DATASET_REGISTRY) {
  cat(sprintf("\n>>> %s\n", ds$name))
  DATASET <- tryCatch(ds$loader(), error = function(e) {
    message(sprintf("  [SKIP] %s: %s", ds$name, conditionMessage(e)))
    NULL
  })
  if (is.null(DATASET) || nrow(DATASET) < 80) {
    cat(sprintf("  [SKIP] n=%s\n",
                if (is.null(DATASET)) "NA" else nrow(DATASET)))
    next
  }

  PRED_NAMES <- setdiff(names(DATASET), "y")
  non_num    <- PRED_NAMES[!sapply(DATASET[, PRED_NAMES, drop = FALSE], is.numeric)]
  if (length(non_num) > 0) {
    cat(sprintf("  [SKIP] non-numeric predictors: %s\n",
                paste(non_num, collapse = ", ")))
    next
  }
  cat(sprintf("  n=%d  p=%d  pos=%d  neg=%d\n",
              nrow(DATASET), length(PRED_NAMES),
              sum(DATASET$y == "pos"), sum(DATASET$y == "neg")))

  for (mech in MISS_MECHS) {
    cat(sprintf("  [%s] %s:", ds$name, mech))
    for (miss_rate in MISS_RATES) {
      cat(sprintf(" %g%%", miss_rate * 100))
      for (sp in seq_len(N_SPLITS)) {
        sp_seed <- sp * 137L +
          as.integer(miss_rate * 1000) +
          nchar(ds$name)
        set.seed(sp_seed)

        tr_idx <- caret::createDataPartition(
          DATASET$y, p = TRAIN_FRAC, list = FALSE)[, 1]
        te_idx <- setdiff(seq_len(nrow(DATASET)), tr_idx)

        X_tr <- DATASET[tr_idx, PRED_NAMES, drop = FALSE]
        y_tr <- DATASET[tr_idx, "y"]
        X_te <- DATASET[te_idx, PRED_NAMES, drop = FALSE]
        y_te <- DATASET[te_idx, "y"]

        # Complete-data reference
        base <- do.call(rbind, lapply(CLF_NAMES, function(cl) {
          mt <- compute_metrics(fit_clf(cl, X_tr, y_tr, X_te), y_te)
          data.frame(Method = "Complete", Classifier = cl, t(mt),
                     stringsAsFactors = FALSE)
        }))
        base$Dataset    <- ds$name; base$MissMech <- mech
        base$MissRate   <- 0;       base$Split    <- sp
        base$ActualMiss <- 0
        all_results[[length(all_results) + 1]] <- base

        res_pair <- tryCatch(
          run_one(mech, miss_rate, sp_seed, X_tr, y_tr, X_te, y_te),
          error = function(e) {
            message(sprintf("\n  [ERR %s sp=%d mr=%.0f%%] %s",
                            ds$name, sp, miss_rate * 100, conditionMessage(e)))
            NULL
          })

        if (!is.null(res_pair)) {
          res             <- res_pair$scores
          res$Dataset     <- ds$name;   res$MissMech  <- mech
          res$MissRate    <- miss_rate; res$Split      <- sp
          res$ActualMiss  <- round(mean(is.na(
            induce_missing(X_tr, miss_rate, mech, seed = sp_seed))), 4)
          all_results[[length(all_results) + 1]] <- res

          tmg <- res_pair$timing
          if (!is.null(tmg) && nrow(tmg) > 0) {
            tmg$Dataset  <- ds$name;   tmg$MissMech <- mech
            tmg$MissRate <- miss_rate; tmg$Split     <- sp
            all_timings[[length(all_timings) + 1]] <- tmg
          }

          vc <- res_pair$varcomp
          if (!is.null(vc) && nrow(vc) > 0) {
            vc$Dataset  <- ds$name;   vc$MissMech <- mech
            vc$MissRate <- miss_rate; vc$Split     <- sp
            all_varcomp[[length(all_varcomp) + 1]] <- vc
          }
        }
        cat(".")
      }
    }
    cat(" done\n")
  }
}
cat("\n=== All runs complete ===\n")


# =============================================================================
# 12. SAVE RESULTS
# =============================================================================

combined           <- do.call(rbind, all_results)
rownames(combined) <- NULL
combined$MissRate  <- as.numeric(combined$MissRate)
combined$MissMech  <- factor(combined$MissMech, levels = MISS_MECHS)
combined$Method    <- factor(combined$Method,   levels = METHOD_ORDER)

write.csv(combined,
          file.path(OUTPUT_DIR, "Multi_HAS_results.csv"),
          row.names = FALSE)
cat("Saved: Multi_HAS_results.csv\n")


# =============================================================================
# 13. SUMMARIES
# =============================================================================

per_ds_summ <- combined %>%
  dplyr::group_by(Dataset, MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(
    AUC_mean   = round(mean(AUC,   na.rm = TRUE), 4),
    AUC_sd     = round(sd(AUC,     na.rm = TRUE), 4),
    Brier_mean = round(mean(Brier, na.rm = TRUE), 4),
    Brier_sd   = round(sd(Brier,   na.rm = TRUE), 4),
    .groups = "drop")

grand_summ <- combined %>%
  dplyr::group_by(MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(
    AUC_mean   = round(mean(AUC,   na.rm = TRUE), 4),
    AUC_sd     = round(sd(AUC,     na.rm = TRUE), 4),
    Brier_mean = round(mean(Brier, na.rm = TRUE), 4),
    Brier_sd   = round(sd(Brier,   na.rm = TRUE), 4),
    .groups = "drop")

write.csv(per_ds_summ,
          file.path(OUTPUT_DIR, "Multi_HAS_per_dataset_summary.csv"),
          row.names = FALSE)
write.csv(grand_summ,
          file.path(OUTPUT_DIR, "Multi_HAS_grand_summary.csv"),
          row.names = FALSE)
cat("Saved: summaries\n")


# =============================================================================
# 14. WILCOXON SIGNED-RANK TESTS
# =============================================================================

INNER_FAMILY_METHODS <- c("MeanSI", "KNN", "RFI", "MIE-SE")
n_comp <- length(INNER_FAMILY_METHODS) * length(CLF_NAMES)  # Bonferroni: 20

ds_level_wt <- combined %>%
  dplyr::group_by(Dataset, MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(AUC_ds   = mean(AUC,   na.rm = TRUE),
                   Brier_ds = mean(Brier, na.rm = TRUE),
                   .groups  = "drop")

compute_stack_wilcoxon <- function(stack_method) {
  rows_local <- list()
  stack_ds   <- ds_level_wt %>%
    dplyr::filter(Method == stack_method, Classifier == "All") %>%
    dplyr::select(Dataset, MissMech, MissRate,
                  AUC_stack = AUC_ds, Brier_stack = Brier_ds)
  if (nrow(stack_ds) == 0) return(NULL)

  for (meth in INNER_FAMILY_METHODS) {
    for (cl in CLF_NAMES) {
      comp_ds  <- ds_level_wt %>%
        dplyr::filter(Method == meth, Classifier == cl) %>%
        dplyr::select(Dataset, MissMech, MissRate,
                      AUC_comp = AUC_ds, Brier_comp = Brier_ds)
      if (nrow(comp_ds) == 0) next

      paired_all <- dplyr::inner_join(stack_ds, comp_ds,
                                      by = c("Dataset", "MissMech", "MissRate"))

      for (mech in MISS_MECHS) {
        for (mr in MISS_RATES) {
          paired <- paired_all %>%
            dplyr::filter(MissMech == mech, abs(MissRate - mr) < 1e-6)
          if (nrow(paired) < 2) next

          for (metric in c("AUC", "Brier")) {
            stack_col <- if (metric == "AUC") paired$AUC_stack else paired$Brier_stack
            comp_col  <- if (metric == "AUC") paired$AUC_comp  else paired$Brier_comp
            diff_vec  <- stack_col - comp_col
            n         <- sum(!is.na(diff_vec))
            if (n < 2) next

            wt    <- tryCatch(
              wilcox.test(stack_col, comp_col, paired = TRUE, exact = FALSE),
              error = function(e) NULL)
            if (is.null(wt)) next

            p_adj  <- round(min(wt$p.value * n_comp, 1), 5)
            better <- if (metric == "AUC") diff_vec > 0 else diff_vec < 0
            worse  <- if (metric == "AUC") diff_vec < 0 else diff_vec > 0

            rows_local[[length(rows_local) + 1]] <- data.frame(
              StackMethod   = stack_method,
              Metric        = metric,
              MissMech      = mech,
              MissRate      = mr,
              Competitor    = meth,
              CompetitorClf = cl,
              n_pairs       = n,
              W             = round(wt$statistic, 1),
              p_value       = round(wt$p.value, 5),
              p_adj_Bonf    = p_adj,
              Signif        = dplyr::case_when(
                p_adj < 0.001 ~ "***",
                p_adj < 0.01  ~ "**",
                p_adj < 0.05  ~ "*",
                TRUE          ~ "ns"),
              mean_diff     = round(mean(diff_vec, na.rm = TRUE), 4),
              sd_diff       = round(sd(diff_vec,   na.rm = TRUE), 4),
              Stack_wins    = sum(better, na.rm = TRUE),
              Stack_losses  = sum(worse,  na.rm = TRUE),
              stringsAsFactors = FALSE)
          }
        }
      }
    }
  }
  if (length(rows_local) == 0) return(NULL)
  do.call(rbind, rows_local)
}

wilcox_df <- compute_stack_wilcoxon("MICE-GLMM-Stack")
if (!is.null(wilcox_df)) {
  rownames(wilcox_df) <- NULL
  write.csv(wilcox_df,
            file.path(OUTPUT_DIR, "Multi_HAS_wilcoxon.csv"),
            row.names = FALSE)
  cat("Saved: Multi_HAS_wilcoxon.csv\n")
}


# =============================================================================
# 15. KENDALL'S W RANK TABLES
# =============================================================================

compute_kendall_W <- function(rank_mat) {
  n <- nrow(rank_mat); k <- ncol(rank_mat)
  if (n < 2 || k < 2)
    return(list(W = NA_real_, chi2 = NA_real_,
                df = NA_integer_, p_value = NA_real_))

  R_i   <- rowSums(rank_mat, na.rm = TRUE)
  R_bar <- k * (n + 1) / 2
  SS    <- sum((R_i - R_bar)^2)

  CF_j <- sapply(seq_len(k), function(j) {
    rj  <- rank_mat[!is.na(rank_mat[, j]), j]
    t_g <- as.numeric(table(rj))
    sum(t_g^3 - t_g) / 12
  })

  denom <- (k^2 * (n^3 - n)) / 12 - k * sum(CF_j)
  W     <- if (abs(denom) < 1e-12) NA_real_ else SS / denom
  W     <- if (!is.na(W)) max(0, min(1, W)) else NA_real_
  chi2  <- k * (n - 1) * W
  df    <- as.integer(n - 1)
  pval  <- pchisq(chi2, df = df, lower.tail = FALSE)

  list(W = round(W, 4), chi2 = round(chi2, 3),
       df = df, p_value = signif(pval, 4))
}

build_rank_table <- function(data_in, miss_rate_val, metric = "AUC",
                             higher_is_better = TRUE, mech = "MCAR") {
  sub <- data_in %>%
    dplyr::filter(Method != "Complete",
                  abs(MissRate - miss_rate_val) < 1e-6)
  if (nrow(sub) == 0) return(NULL)

  agg <- sub %>%
    dplyr::group_by(Method, Classifier, Dataset) %>%
    dplyr::summarise(metric_mean = mean(.data[[metric]], na.rm = TRUE),
                     .groups = "drop") %>%
    dplyr::mutate(
      Method    = factor(Method, levels = METHOD_ORDER),
      MethodClf = paste(Method, Classifier, sep = "/"))

  row_order <- c(
    paste("MeanSI",         CLF_NAMES, sep = "/"),
    paste("KNN",            CLF_NAMES, sep = "/"),
    paste("RFI",            CLF_NAMES, sep = "/"),
    paste("CCA",            CLF_NAMES, sep = "/"),
    paste("MICE-PMM",       CLF_NAMES, sep = "/"),
    "MICE-GLMM-Stack/All",
    paste("MIE-SE",         CLF_NAMES, sep = "/")
  )
  row_order <- row_order[row_order %in% unique(agg$MethodClf)]

  wide_auc <- agg %>%
    dplyr::select(MethodClf, Dataset, metric_mean) %>%
    tidyr::pivot_wider(names_from  = Dataset,
                       values_from = metric_mean) %>%
    dplyr::mutate(MethodClf = factor(MethodClf, levels = row_order)) %>%
    dplyr::arrange(MethodClf)

  ds_cols   <- setdiff(names(wide_auc), "MethodClf")
  sign_flip <- if (higher_is_better) -1 else 1

  rank_wide <- wide_auc
  for (dc in ds_cols)
    rank_wide[[dc]] <- rank(sign_flip * wide_auc[[dc]],
                            ties.method = "average", na.last = "keep")

  rank_mat           <- as.matrix(rank_wide[, ds_cols, drop = FALSE])
  rownames(rank_mat) <- as.character(rank_wide$MethodClf)
  rank_wide$MeanRank    <- round(rowMeans(rank_mat, na.rm = TRUE), 2)
  rank_wide$OverallRank <- rank(rank_wide$MeanRank, ties.method = "average")
  rank_wide             <- rank_wide[order(rank_wide$OverallRank), ]

  kw <- compute_kendall_W(as.matrix(rank_wide[, ds_cols, drop = FALSE]))

  list(miss_rate = miss_rate_val, metric = metric, mech = mech,
       rank_wide = rank_wide, datasets = ds_cols, kendall = kw)
}

cat("\n=== Kendall W Rank Tables ===\n")

RANK_METRICS <- list(AUC = TRUE, Brier = FALSE)

for (mech in MISS_MECHS) {
  mech_data <- combined %>% dplyr::filter(MissMech == mech)
  for (met in names(RANK_METRICS)) {
    for (mr in MISS_RATES) {
      rt <- build_rank_table(mech_data, mr,
                             metric           = met,
                             higher_is_better = RANK_METRICS[[met]],
                             mech             = mech)
      if (is.null(rt)) next
      kw <- rt$kendall
      cat(sprintf("[%s | %s | %d%%] W=%.4f  chi2=%.3f  df=%d  p=%s\n",
                  mech, met, as.integer(mr * 100), kw$W, kw$chi2, kw$df,
                  ifelse(kw$p_value < 0.001, "<0.001",
                         sprintf("%.4f", kw$p_value))))

      csv_file <- file.path(OUTPUT_DIR,
        sprintf("Multi_HAS_rank_%s_%02dpct_%s.csv", mech, as.integer(mr * 100), met))
      write.csv(rt$rank_wide, csv_file, row.names = FALSE)
    }
  }
}


# =============================================================================
# 16. VARIANCE COMPONENT SUMMARY
# =============================================================================

varcomp_combined <- do.call(rbind, all_varcomp)
if (!is.null(varcomp_combined) && nrow(varcomp_combined) > 0) {
  rownames(varcomp_combined) <- NULL
  write.csv(varcomp_combined,
            file.path(OUTPUT_DIR, "Multi_VarianceComponents_Raw.csv"),
            row.names = FALSE)
  cat("Saved: Multi_VarianceComponents_Raw.csv\n")
}


# =============================================================================
# 17. RUNTIME SUMMARY
# =============================================================================

timing_combined <- do.call(rbind, all_timings)
if (!is.null(timing_combined) && nrow(timing_combined) > 0) {
  rownames(timing_combined) <- NULL
  write.csv(timing_combined,
            file.path(OUTPUT_DIR, "Multi_Runtime_Raw.csv"),
            row.names = FALSE)

  runtime_summary <- timing_combined %>%
    dplyr::group_by(MissMech, Method) %>%
    dplyr::summarise(
      N        = dplyr::n(),
      Mean_sec = round(mean(RunTime_sec, na.rm = TRUE), 4),
      SD_sec   = round(sd(RunTime_sec,   na.rm = TRUE), 4),
      Min_sec  = round(min(RunTime_sec,  na.rm = TRUE), 4),
      Max_sec  = round(max(RunTime_sec,  na.rm = TRUE), 4),
      .groups  = "drop")

  write.csv(runtime_summary,
            file.path(OUTPUT_DIR, "Multi_Runtime_Summary.csv"),
            row.names = FALSE)
  cat("Saved: Multi_Runtime_Raw.csv, Multi_Runtime_Summary.csv\n")
}


# =============================================================================
# 18. VARIANCE COMPONENTS FIGURE (Figure 1)
# =============================================================================

library(patchwork)

vc_data <- tibble::tribble(
  ~Component,                       ~Mech,  ~Mean_vcov,  ~Mean_sdcor,
  "Classifier — slope (logit p)",   "MCAR",  0.16109,     0.38301,
  "Classifier — intercept",         "MCAR",  0.10375,     0.25408,
  "Classifier — slope/int. corr.",  "MCAR",  0.04466,     0.34256,
  "Method intercept (crossed)",     "MCAR",  0.00015,     0.00478,
  "Draw intercept (nested)",        "MCAR",  0.00000,     0.00000,
  "Classifier — slope (logit p)",   "MAR",   0.16420,     0.38698,
  "Classifier — intercept",         "MAR",   0.10531,     0.25579,
  "Classifier — slope/int. corr.",  "MAR",   0.04233,     0.34948,
  "Method intercept (crossed)",     "MAR",   0.00011,     0.00396,
  "Draw intercept (nested)",        "MAR",   0.00000,     0.00000
)

clf_comp <- c("Classifier — slope (logit p)",
              "Classifier — intercept",
              "Classifier — slope/int. corr.")
imp_comp <- c("Method intercept (crossed)",
              "Draw intercept (nested)")

vc_data <- vc_data %>%
  dplyr::mutate(
    Component = factor(Component, levels = c(clf_comp, imp_comp)),
    Mech      = factor(Mech, levels = c("MCAR", "MAR")))

col_mcar <- "#2166AC"
col_mar  <- "#1A9850"

th_fig <- function(bs = 9) {
  theme_classic(base_size = bs, base_family = "Arial") +
    theme(
      axis.line          = element_line(linewidth = 0.35, colour = "grey30"),
      axis.ticks         = element_line(linewidth = 0.30, colour = "grey30"),
      axis.text          = element_text(size = rel(0.88), colour = "grey15"),
      axis.title.y       = element_text(size = rel(0.92), margin = margin(r = 4)),
      axis.title.x       = element_blank(),
      panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.25),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      legend.position    = "top",
      legend.title       = element_blank(),
      legend.text        = element_text(size = rel(0.88)),
      legend.key.size    = unit(0.55, "lines"),
      plot.title         = element_text(face = "bold", size = rel(1.0), hjust = 0),
      plot.margin        = margin(5, 8, 3, 5)
    )
}

pA <- vc_data %>%
  dplyr::filter(Component %in% clf_comp) %>%
  ggplot(aes(x = Component, y = Mean_sdcor, fill = Mech)) +
  geom_col(position = position_dodge(0.70), width = 0.60,
           colour = "white", linewidth = 0.15) +
  geom_text(aes(label = sprintf("%.3f", Mean_sdcor)),
            position = position_dodge(0.70),
            vjust = -0.40, size = 2.3, colour = "grey20") +
  scale_fill_manual(values = c(MCAR = col_mcar, MAR = col_mar)) +
  scale_x_discrete(labels = c(
    "Classifier — slope (logit p)"  = "Slope\n(logit p)",
    "Classifier — intercept"        = "Intercept",
    "Classifier — slope/int. corr." = "Slope/intercept\ncorrelation")) +
  scale_y_continuous(limits = c(0, 0.46), breaks = seq(0, 0.4, 0.1),
                     expand = expansion(mult = c(0, 0.06))) +
  labs(title = "(A)  Classifier random effects",
       y     = "Estimated SD (logit link scale)") +
  th_fig()

pB <- vc_data %>%
  dplyr::filter(Component %in% imp_comp) %>%
  dplyr::mutate(sdcor_plot = ifelse(Mean_sdcor == 0, 0.00003, Mean_sdcor)) %>%
  ggplot(aes(x = Component, y = sdcor_plot, fill = Mech)) +
  geom_col(position = position_dodge(0.70), width = 0.60,
           colour = "white", linewidth = 0.15) +
  geom_text(aes(label = sprintf("%.5f", Mean_sdcor)),
            position = position_dodge(0.70),
            vjust = -0.40, size = 2.3, colour = "grey20") +
  annotate("text", x = "Draw intercept (nested)", y = 0.0048,
           label = "Singular fit\n(boundary estimate)",
           size = 2.1, colour = "grey50", hjust = 0.5, lineheight = 1.1) +
  scale_fill_manual(values = c(MCAR = col_mcar, MAR = col_mar)) +
  scale_x_discrete(labels = c(
    "Method intercept (crossed)" = "Method intercept\n(pmm/cart/norm/rf)",
    "Draw intercept (nested)"    = "Draw intercept\n(nested in method)")) +
  scale_y_continuous(limits = c(0, 0.0073), breaks = seq(0, 0.006, 0.002),
                     labels = function(x) sprintf("%.3f", x),
                     expand = expansion(mult = c(0, 0.06))) +
  labs(title = "(B)  Imputation random effects",
       y     = "Estimated SD (logit link scale)") +
  th_fig()

bar_cols <- c(
  "Clf slope"             = "#2166AC",
  "Clf intercept"         = "#4393C3",
  "Clf slope/int. corr."  = "#92C5DE",
  "Method"                = "#1A9850",
  "Draw"                  = "#A6D96A"
)

prop_data <- vc_data %>%
  dplyr::group_by(Mech) %>%
  dplyr::mutate(Total = sum(Mean_vcov)) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    Share = Mean_vcov / Total * 100,
    Lbl   = dplyr::case_when(
      Component == "Classifier — slope (logit p)"  ~ "Clf slope",
      Component == "Classifier — intercept"        ~ "Clf intercept",
      Component == "Classifier — slope/int. corr." ~ "Clf slope/int. corr.",
      Component == "Method intercept (crossed)"    ~ "Method",
      TRUE                                         ~ "Draw"),
    Lbl = factor(Lbl, levels = names(bar_cols)))

pC <- ggplot(prop_data, aes(x = Mech, y = Share, fill = Lbl)) +
  geom_col(width = 0.52, colour = "white", linewidth = 0.2) +
  geom_text(data = dplyr::filter(prop_data, Share > 2),
            aes(label = sprintf("%.1f%%", Share)),
            position = position_stack(vjust = 0.5),
            size = 2.3, colour = "white", fontface = "bold") +
  scale_fill_manual(values = bar_cols,
                    guide  = guide_legend(nrow = 1)) +
  scale_y_continuous(breaks = seq(0, 100, 20),
                     labels = function(x) paste0(x, "%"),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(title = "(C)  Proportional share of total variance",
       y     = "Share (%)", fill = NULL) +
  th_fig() +
  theme(legend.position = "bottom",
        legend.text     = element_text(size = rel(0.82)))

fig_vc <- (pA | pB) / pC +
  plot_layout(heights = c(1, 0.95)) &
  theme(legend.position = "top",
        text = element_text(family = "Arial"))

W_in <- 190 / 25.4
H_in <- 160 / 25.4

ggsave(file.path(OUTPUT_DIR, "fig_variance_clean.pdf"),
       plot = fig_vc, width = W_in, height = H_in,
       units = "in", device = cairo_pdf)
ggsave(file.path(OUTPUT_DIR, "fig_variance_clean.tiff"),
       plot = fig_vc, width = W_in, height = H_in,
       units = "in", dpi = 600, compression = "lzw")
ggsave(file.path(OUTPUT_DIR, "fig_variance_clean.png"),
       plot = fig_vc, width = W_in, height = H_in,
       units = "in", dpi = 900)
cat("Saved: fig_variance_clean (.pdf / .tiff / .png)\n")

cat("\n=== DONE ===\n")
