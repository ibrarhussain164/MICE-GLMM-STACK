# =============================================================================
# Multi_HAS_Pipeline.R — 15 Numeric-Predictor Binary Clinical / Epidemiological Datasets
# MICE-GLMM-Stack: pooled 3-random-effect GLMM stacking benchmark
# =============================================================================
#
# INCLUSION CRITERIA:
#   [OK] Binary outcome (no censoring, no survival time)
#   [OK] All retained predictors are numeric-valued; nominal, ordinal, and
#     binary-coded categorical predictors were excluded
#   [OK] Clinical or epidemiological domain
#
# MIE-SE uses MICE-PMM exclusively (all predictors numeric-valued).
# MICE-GLMM-Stack internally pools 3 more MICE base methods (cart, norm, rf)
# in addition to pmm -- see Sec.11b -- but only MICE-PMM is reported as its own
# standalone competitive row; MICE-CART/Norm/RF remain internal-only inputs
# to the pooled stack (midastouch dropped -- see Sec.9c.)
# (MIE_PMM_AVG removed: it duplicated MICE-PMM's PMM-averaging procedure --
# same imputation method, same 5-draw average -- differing only by RNG seed,
# with no separate analytical purpose of its own.)

# METHODS:
#   Complete        : complete-data reference (no missing, no imputation)
#   MeanSI          : mean single imputation
#   KNN             : VIM::kNN single imputation
#   RFI             : missForest single imputation
#   CCA             : complete case analysis (listwise deletion of training
#                  rows with any missing predictor cell; no imputation)
#   MICE-PMM        : individual MICE base-method result, per classifier
#                  (m=5 draws, averaged) -- the only individual MICE base
#                  method reported as a standalone competitor; MICE-GLMM-Stack
#                  internally also pools cart/norm/rf (see below) but those
#                  are not reported as their own rows (midastouch dropped
#                  from the whole pipeline -- see Sec.9c)
#   MICE-GLMM-Stack : pooled 3-random-effect GLMM meta-learner spanning
#                  ALL 5 classifiers (J48/NB/PART/SMO/RF) x 4 MICE base
#                  methods (pmm/cart/norm/rf) x M_IMP_STACK=5 draws each =
#                  100 OOF/test probability streams per split (midastouch
#                  dropped -- redundant with pmm's donor-matching family, and
#                  the most compute-heavy of the donor methods; removing it
#                  preserves coverage of every imputation paradigm --
#                  matching/parametric/single-tree/ensemble -- while cutting
#                  cost), modelling THREE sources of variation:
#                    (1) base classifier   J48/NB/PART/SMO/RF  (random slope)
#                    (2) MICE base method  pmm/cart/norm/rf
#                    (3) imputation draw   nested within method (m=1..5)
#                  combined via:
#                    y_bin ~ logit_p + (logit_p | clf_id) + (1 | method_id/draw_id)
#                  with a fallback formula ladder if the full model fails to
#                  converge (see Sec.9c). Reported ONCE per split as
#                  Classifier = "All" (pooled across classifiers and methods).
#   MIE-SE      : Paper-style stacking -- Aleryani (2020); per meta-clf:
#                  MIE-SE/J48, MIE-SE/NB, MIE-SE/PART, MIE-SE/SMO, MIE-SE/RF
#
# NOTE: MICE-GLMM-Stack reports Classifier = "All" only (one pooled value
# per split). The CCA and MICE-PMM rows remain per classifier, as does
# MIE-SE.
#
# COMPARISON DESIGN (Sec.16): The pooled MICE-GLMM-Stack/All result is compared
# via paired Wilcoxon signed-rank tests against every (imputation method x
# classifier) combination -- MeanSI, KNN, RFI, CCA, MICE-PMM, and MIE-SE, each
# evaluated with J48/NB/PART/SMO/RF (6 x 5 = 30 competitors). Bonferroni
# correction is applied within each (MissRate x Metric) family (n_comp = 30).
# This yields 5 x 2 = 10 tables, each with 30 competitor rows.
#
# LITERATURE BASIS FOR THE STACKING GLMM (MICE-GLMM-Stack Sec.9c):
#   - Rubin (1987, "Multiple Imputation for Nonresponse in Surveys", Wiley)
#     and van Buuren (2018, "Flexible Imputation of Missing Data", 2nd ed.,
#     CRC Press) establish that between-imputation variability is a genuine,
#     estimable source of uncertainty that should be modelled explicitly
#     rather than discarded by simple averaging across draws -- motivating a
#     random "imp_id" (draw) effect rather than pre-averaging the m=5 draws.
#   - Aleryani, Wang & De La Iglesia (2020, the MIE-SE stacking design used
#     elsewhere in this script), Wolpert (1992, "Stacked Generalization",
#     Neural Networks 5(2)), and van der Laan, Polley & Hubbard (2007, "Super
#     Learner", Statistical Applications in Genetics and Molecular Biology
#     6(1)) justify combining several base classifiers' predictions through a
#     learned meta-model rather than unweighted averaging -- motivating the
#     crossed "classifier" random effect, with a random slope on logit_p
#     since classifiers differ systematically in calibration, not only mean
#     score.
#   - Gelman & Hill (2007, "Data Analysis Using Regression and Multilevel/
#     Hierarchical Models", Cambridge UP, esp. Ch.13) give the standard
#     guidance for preferring varying-slope + varying-intercept terms when
#     group-specific calibration is expected -- i.e. (logit_p | clf_id) -- and
#     a simple random intercept for the imputation-draw source
#     (1 | method_id/draw_id).
#   - Bates, Machler, Bolker & Walker (2015, "Fitting Linear Mixed-Effects
#     Models Using lme4", J. Statistical Software 67(1)) document the
#     glmer() syntax and fallback strategies used below when the full
#     varying-slope model fails to converge.
#   - The MICE base method itself differs systematically in imputation
#     quality (van Buuren 2018, Ch.3 on method choice), justifying a random
#     intercept for "method_id", with imputation draws nested within method
#     (1 | method_id/draw_id) since draws are only exchangeable within the
#     same method, not across methods -- combined with the crossed classifier
#     random slope above into y_bin ~ logit_p + (logit_p | clf_id) +
#     (1 | method_id/draw_id).
#
# MISSING MECHANISM : MCAR (primary, following original MIE framework) AND
#                      MAR (severity-weighted, clinically-motivated; see
#                      induce_missing() in Sec.3 for design + citations)
# CLASSIFIERS       : J48, NB, PART, SMO, RF
# MISS RATES        : 10 20 30 40 50 %  (predictor-cell level; bycases=FALSE)
# PRIMARY METRIC    : AUC (threshold-independent, robust to class imbalance)
# SECONDARY METRIC  : Brier score (probability calibration)
# SPLITS            : 10 stratified 70/30 (caret::createDataPartition)
# M_IMP             : 5  (draws per MICE base method)
# MAXIT_MICE        : 20
# =============================================================================
# --- 0. Packages ---------------------------------------------------------
suppressPackageStartupMessages({
  library(mice); library(VIM); library(missForest)
  library(C50); library(naivebayes); library(e1071); library(randomForest)
  library(rpart)   # required by mice's "cart" imputation method
  library(lme4); library(pROC); library(caret)
  library(dplyr); library(tidyr); library(ggplot2); library(purrr)
})
set.seed(2024)
# --- 1. Settings -----------------------------------------------------------
N_SPLITS    <- 10
MISS_RATES  <- c( 0.1, 0.2, 0.3, 0.40, 0.50)
M_IMP       <- 5
M_IMP_SE    <- 5
MAXIT_MICE  <- 20
CV_FOLDS    <- 10
TRAIN_FRAC  <- 0.70
CLF_NAMES   <- c("J48", "NB", "PART", "SMO", "RF")
MISS_MECHS  <- c("MCAR", "MAR")
mech_labels <- c(MCAR = "MCAR -- Completely At Random",
                 MAR  = "MAR -- Missing At Random (severity-weighted)")
# --- MICE base-method settings ---------------------------------------------
MICE_STACK_METHODS <- c("pmm", "cart", "norm", "rf")
M_IMP_STACK         <- M_IMP   # draws per base method (5), per user confirmation
MICE_METHOD_LABEL   <- c(pmm = "MICE-PMM", cart = "MICE-CART", norm = "MICE-Norm",
                         rf = "MICE-RF")
METHOD_ORDER <- c("Complete", "MeanSI", "KNN", "RFI", "CCA",
                  "MICE-PMM",
                  "MICE-GLMM-Stack", "MIE-SE")
OUTPUT_DIR  <- "~/Desktop/project4/GLMM-stack/GLMM-stack/GLMM-stack-mice"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
DATA_DIR    <- "~/Desktop/project4/GLMM-stack/data"
# --- 2. Dataset Loaders ------------------------------------------------------
read_ds <- function(name) {
  d <- read.csv(file.path(DATA_DIR, paste0(name, ".csv")),
                stringsAsFactors = FALSE)
  
  if (!"y" %in% names(d)) {
    candidates <- c("y", "Y", "outcome", "Outcome", "OUTCOME", "target",
                    "Target", "TARGET", "class", "Class", "CLASS",
                    "label", "Label", "LABEL", "DEATH_EVENT", "death_event",
                    "Death_Event", "diagnosis", "Diagnosis", "DIAGNOSIS",
                    "status", "Status", "STATUS", "event", "Event", "EVENT")
    found <- intersect(candidates, names(d))
    if (length(found) == 0)
      stop(sprintf(
        "[%s] no outcome column found (looked for one of: %s; CSV columns are: %s).",
        name, paste(candidates, collapse = ", "), paste(names(d), collapse = ", ")))
    names(d)[names(d) == found[1]] <- "y"
    cat(sprintf("[%s] outcome column '%s' renamed to 'y'\n", name, found[1]))
  }
  
  time_cols <- intersect(c("time", "Time", "TIME", "followup_time",
                           "follow_up_time", "FollowUpTime"), names(d))
  if (length(time_cols) > 0) {
    d <- d[, !(names(d) %in% time_cols), drop = FALSE]
    cat(sprintf("[%s] dropped survival-time column(s): %s\n",
                name, paste(time_cols, collapse = ", ")))
  }
  
  y_orig    <- d$y
  y_chr_raw <- toupper(trimws(as.character(y_orig)))
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
    warning(sprintf(
      "[%s] outcome value(s) %s could not be mapped to neg/pos and will be dropped as missing -- verify the raw outcome coding for this dataset.",
      name, paste(unmapped, collapse = ", ")), call. = FALSE)
  
  n0 <- nrow(d)
  d  <- d[stats::complete.cases(d), , drop = FALSE]
  n1 <- nrow(d)
  if (n1 < n0)
    cat(sprintf("[%s] dropped %d/%d rows with original (pre-existing) NA\n",
                name, n0 - n1, n0))
  
  tab <- table(d$y)
  if (length(tab) < 2 || any(tab < 2))
    stop(sprintf(
      paste0("[%s] outcome y is not usably binary after coercion (counts: %s). ",
             "Check the raw outcome coding for this dataset -- it may use a ",
             "convention (e.g. a 1=absence/2=presence scheme) not covered by ",
             "the automatic neg/pos mapping in read_ds()."),
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
# --- 3. Missing Data Injection -----------------------------------------------
induce_missing <- function(X, rate, mech, seed = NULL) {
  if (rate == 0) return(X)
  mech_up <- toupper(mech)
  
  if (!is.null(seed)) {
    mech_offset <- switch(mech_up, MCAR = 1009L, MAR = 7919L, 4001L)
    set.seed(as.integer(seed) + mech_offset)
  }
  
  p <- ncol(X)
  min_observed <- if (p >= 4L) 2L else 1L
  k_needed <- max(1L, min(p - min_observed, ceiling(rate * p * 1.3) + 1L))
  patterns_mat <- matrix(1L, nrow = p, ncol = p)
  for (i in seq_len(p)) {
    miss_cols <- ((i - 1L) + 0L:(k_needed - 1L)) %% p + 1L
    patterns_mat[i, miss_cols] <- 0L
  }
  
  amp_err <- NULL
  amp_result <- suppressMessages(suppressWarnings(tryCatch({
    if (mech_up == "MAR") {
      case_frac <- min(0.999, rate * p / k_needed)
      if (case_frac <= 0.5) {
        odds_ratio <- 20
      } else {
        x_sat <- 0.5 / case_frac
        odds_ratio <- max(1.01, min(20, x_sat / (1 - x_sat)))
      }
      strong_odds <- matrix(rep(c(1, odds_ratio), p), nrow = p, byrow = TRUE)
      amp <- mice::ampute(X, prop = rate, mech = "MAR", bycases = FALSE,
                          patterns = patterns_mat, cont = FALSE, odds = strong_odds)
    } else {
      amp <- mice::ampute(X, prop = rate, mech = mech_up, bycases = FALSE,
                          patterns = patterns_mat)
    }
    amp$amp
  }, error = function(e) { amp_err <<- conditionMessage(e); NULL })))
  
  if (!is.null(amp_result)) return(amp_result)
  
  message(sprintf("  [induce_missing] ampute() failed for mech=%s: %s -- using manual fallback",
                  mech_up, if (is.null(amp_err)) "<unknown error>" else amp_err))
  n_cells_total <- nrow(X) * ncol(X)
  n_miss_total  <- max(1L, floor(rate * n_cells_total))
  X_miss <- X
  if (mech_up == "MAR") {
    z <- scale(as.matrix(X))
    z[!is.finite(z)] <- 0
    severity  <- rowMeans(z)
    rank_pct  <- rank(severity, ties.method = "average") / length(severity)
    odds_row  <- 1 + 19 * rank_pct
    p_row     <- odds_row / sum(odds_row)
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
# --- 4. Imputation ---------------------------------------------------------
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
# --- NA FIX HELPER -- applies to every MICE base method (pmm/cart/norm/rf) ---
fix_residual_nas <- function(df) {
  for (nm in names(df)) {
    if (any(is.na(df[[nm]]))) {
      fv <- mean(df[[nm]], na.rm = TRUE)
      if (is.na(fv)) fv <- 0
      df[[nm]][is.na(df[[nm]])] <- fv
    }
  }
  df
}
mice_impute <- function(X_miss, m = M_IMP, y_vec = NULL,
                        seed = 1, maxit = MAXIT_MICE) {
  df <- if (!is.null(y_vec)) data.frame(X_miss, y = y_vec) else X_miss
  meths <- setNames(rep("pmm", ncol(df)), names(df))
  meths["y"] <- ""
  for (nm in names(df)) if (!any(is.na(df[[nm]]))) meths[nm] <- ""
  suppressMessages(suppressWarnings(
    mids <- mice::mice(df, m = m, method = meths,
                       seed = seed, printFlag = FALSE, maxit = maxit)
  ))
  lapply(seq_len(m), function(i) {
    comp <- mice::complete(mids, i)
    comp <- comp[, setdiff(names(comp), "y"), drop = FALSE]
    fix_residual_nas(comp)
  })
}
# --- 4b. MICE imputation by an arbitrary single method (for the 4 individual
# MICE base-method rows) --------------------------------------------------
mice_impute_method <- function(X_miss, method_name, m = M_IMP_STACK, y_vec = NULL,
                               seed = 1, maxit = MAXIT_MICE) {
  df <- if (!is.null(y_vec)) data.frame(X_miss, y = y_vec) else X_miss
  meths <- setNames(rep(method_name, ncol(df)), names(df))
  meths["y"] <- ""
  for (nm in names(df)) if (!any(is.na(df[[nm]]))) meths[nm] <- ""
  suppressMessages(suppressWarnings(
    mids <- mice::mice(df, m = m, method = meths,
                       seed = seed, printFlag = FALSE, maxit = maxit)
  ))
  lapply(seq_len(m), function(i) {
    comp <- mice::complete(mids, i)
    comp <- comp[, setdiff(names(comp), "y"), drop = FALSE]
    fix_residual_nas(comp)
  })
}
# --- 5. Metrics (fixed threshold = 0.5) -------------------------------------
compute_metrics <- function(probs, labels, thresh = 0.5) {
  probs  <- pmin(pmax(as.numeric(probs), 0), 1)
  lb     <- as.integer(labels == "pos")
  auc    <- tryCatch(as.numeric(pROC::auc(pROC::roc(lb, probs, quiet = TRUE))),
                     error = function(e) NA_real_)
  br     <- mean((probs - lb)^2, na.rm = TRUE)
  pred_p <- (probs >= thresh); act_p <- (lb == 1)
  tp <- sum( pred_p &  act_p); fp <- sum( pred_p & !act_p)
  tn <- sum(!pred_p & !act_p); fn <- sum(!pred_p &  act_p)
  se  <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  sp  <- if ((tn + fp) == 0) NA_real_ else tn / (tn + fp)
  c(AUC = auc, Brier = br,
    Sensitivity = se, Specificity = sp, Threshold = thresh)
}
# --- 6. Classifier Wrapper ---------------------------------------------------
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
      if (length(keep) == 0) stop("SMO: no non-degenerate predictors in this fold")
      m <- e1071::svm(df_tr[, keep, drop = FALSE], df_tr$y,
                      kernel = "linear", probability = TRUE)
      attr(predict(m, df_te[, keep, drop = FALSE], probability = TRUE),
           "probabilities")[, "pos"]
    } else if (clf == "RF") {
      m <- randomForest(y ~ ., df_tr, ntree = 200)
      predict(m, df_te, type = "prob")[, "pos"]
    } else { rep(0.5, nrow(df_te)) }
  }, error = function(e) rep(0.5, nrow(df_te))))
  
  n_te <- nrow(df_te)
  if (is.null(out) || length(out) != n_te) return(rep(0.5, n_te))
  out <- as.numeric(out)
  bad <- !is.finite(out)
  if (any(bad)) out[bad] <- 0.5
  pmin(pmax(out, 0), 1)
}
# --- 8. MIE-SE -- Paper-Style Stacking (Aleryani 2020) ----------------------
mice_se_l0 <- function(imp_tr, y_tr, imp_te,
                       m_se = M_IMP_SE, oof_folds = CV_FOLDS) {
  m_use   <- min(m_se, length(imp_tr))
  n_tr    <- length(y_tr); n_te <- nrow(imp_te[[1]])
  col_nms <- as.vector(outer(CLF_NAMES, seq_len(m_use),
                             FUN = function(c, k) paste0(c, "_m", k)))
  
  L0_tr  <- matrix(0.5, n_tr, length(col_nms), dimnames = list(NULL, col_nms))
  k_use  <- min(oof_folds, n_tr)
  folds  <- caret::createFolds(y_tr, k = k_use, list = TRUE, returnTrain = FALSE)
  for (cl in CLF_NAMES)
    for (k in seq_len(m_use)) {
      nm <- paste0(cl, "_m", k)
      for (fi in folds) {
        tri <- setdiff(seq_len(n_tr), fi)
        L0_tr[fi, nm] <- tryCatch(
          fit_clf(cl, imp_tr[[k]][tri, , drop=FALSE], y_tr[tri], imp_tr[[k]][fi, , drop=FALSE]),
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
  
  L0_tr[is.na(L0_tr)] <- 0.5; L0_te[is.na(L0_te)] <- 0.5
  list(L0_tr = L0_tr, L0_te = L0_te)
}
run_mice_se <- function(L0_tr, L0_te, y_tr, meta_clf) {
  df_tr <- as.data.frame(L0_tr)
  df_te <- as.data.frame(L0_te)
  tryCatch(
    fit_clf(meta_clf, df_tr, y_tr, df_te),
    error = function(e) rowMeans(L0_te, na.rm = TRUE))
}
# --- 9. MICE base-method imputations ----------------------------------------
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
  oof_mat  <- matrix(0.5, n_tr, length(meths), dimnames = list(NULL, meths))
  test_mat <- matrix(0.5, nrow(X_te_full), length(meths), dimnames = list(NULL, meths))
  oof_draws  <- setNames(vector("list", length(meths)), meths)
  test_draws <- setNames(vector("list", length(meths)), meths)
  folds <- caret::createFolds(y_tr, k = cv_folds, list = TRUE, returnTrain = FALSE)
  
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
          fit_clf(clf, draw[tri, , drop = FALSE], y_tr[tri], draw[fi, , drop = FALSE]),
          error = function(e) rep(0.5, length(fi)))
      }
    }
    oof_mat[, mi]      <- rowMeans(oof_draw_mat, na.rm = TRUE)
    oof_draws[[meth]]  <- oof_draw_mat
    
    test_draw_mat <- matrix(NA_real_, nrow(X_te_full), length(imp_draws))
    for (di in seq_along(imp_draws)) {
      test_draw_mat[, di] <- tryCatch(
        fit_clf(clf, imp_draws[[di]], y_tr, X_te_full),
        error = function(e) rep(0.5, nrow(X_te_full)))
    }
    test_mat[, mi]       <- rowMeans(test_draw_mat, na.rm = TRUE)
    test_draws[[meth]]   <- test_draw_mat
  }
  oof_mat[is.na(oof_mat)]   <- 0.5
  test_mat[is.na(test_mat)] <- 0.5
  list(oof = oof_mat, test = test_mat, oof_draws = oof_draws, test_draws = test_draws)
}
# --- 9c. MICE-GLMM-Stack -- pooled 3-random-effect GLMM meta-learner --------
build_pooled_glmm_stack_data <- function(stack_scores_by_clf, y_tr) {
  rows_tr <- list(); rows_te <- list()
  clf_ids <- names(stack_scores_by_clf)
  
  for (cl in clf_ids) {
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
          draw_id   = factor(di, levels = seq_len(M_IMP_STACK)),
          method_id = factor(meth, levels = MICE_STACK_METHODS),
          clf_id    = factor(cl, levels = CLF_NAMES),
          stringsAsFactors = FALSE)
      }
      for (di in seq_len(ncol(test_mat))) {
        rows_te[[length(rows_te) + 1]] <- data.frame(
          p         = test_mat[, di],
          draw_id   = factor(di, levels = seq_len(M_IMP_STACK)),
          method_id = factor(meth, levels = MICE_STACK_METHODS),
          clf_id    = factor(cl, levels = CLF_NAMES),
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
    return(list(pred = rowMeans(do.call(cbind, test_list_fallback), na.rm = TRUE),
                varcomp = NULL, formula_used = 4L))
  long_tr <- long_data$long_tr
  long_te <- long_data$long_te
  
  mk_ctrl <- function(maxfun)
    glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = maxfun),
                 check.conv.singular = .makeCC("ignore", tol = 1e-3))
  
  formula_ladder <- list(
    list(f = y_bin ~ logit_p + (logit_p | clf_id) + (1 | method_id/draw_id),
         ctrl = mk_ctrl(5e5)),
    list(f = y_bin ~ logit_p + (1 | clf_id) + (1 | method_id/draw_id),
         ctrl = mk_ctrl(4e5)),
    list(f = y_bin ~ logit_p + (1 | clf_id) + (1 | method_id) + (1 | draw_id),
         ctrl = mk_ctrl(3e5))
  )
  
  ok <- FALSE; mod <- NULL; formula_used <- NA_integer_
  for (fi in seq_along(formula_ladder)) {
    if (ok) break
    fc <- formula_ladder[[fi]]
    suppressMessages(suppressWarnings(tryCatch({
      mod <- lme4::glmer(fc$f, data = long_tr, family = binomial(),
                         control = fc$ctrl)
      ok <- TRUE
      formula_used <- fi
    }, error = function(e) NULL)))
  }
  if (!ok) {
    agg_fb <- aggregate(p ~ obs_id, long_te[, c("obs_id", "p")], mean)
    return(list(pred = agg_fb[order(agg_fb$obs_id), "p"],
                varcomp = NULL, formula_used = 4L))
  }
  long_te$pred <- tryCatch(
    predict(mod, long_te, type = "response", allow.new.levels = TRUE, re.form = NULL),
    error = function(e)
      plogis(predict(mod, long_te, type = "link", allow.new.levels = TRUE, re.form = NULL)))
  agg <- aggregate(pred ~ obs_id, long_te, mean)
  
  vc_df <- tryCatch({
    vc_raw <- as.data.frame(lme4::VarCorr(mod))
    vc_raw[, c("grp", "var1", "var2", "vcov", "sdcor")]
  }, error = function(e) NULL)
  
  list(pred = agg[order(agg$obs_id), "pred"], varcomp = vc_df, formula_used = formula_used)
}
# --- 12. Single Run ----------------------------------------------------------
run_one <- function(miss_mech, miss_rate, split_seed,
                    X_tr_full, y_tr, X_te_full, y_te) {
  rows <- list()
  varcomp_local <- list()
  rec  <- function(method, clf, test_probs) {
    mt <- compute_metrics(test_probs, y_te)
    data.frame(Method = method, Classifier = clf, t(mt), stringsAsFactors = FALSE)
  }
  
  timing_local <- list()
  add_timing <- function(method_label, secs) {
    timing_local[[length(timing_local) + 1]] <<- data.frame(
      Method = method_label, RunTime_sec = as.numeric(secs),
      stringsAsFactors = FALSE)
  }
  X_tr_miss <- tryCatch(induce_missing(X_tr_full, miss_rate, miss_mech, seed = split_seed),
                        error = function(e) X_tr_full)
  actual_miss <- mean(is.na(X_tr_miss))
  attr(X_tr_miss, "actual_miss_rate") <- actual_miss
  
  si_mean <- impute_mean(X_tr_miss, X_te_full)
  si_knn  <- tryCatch(impute_knn(X_tr_miss, X_te_full),
                      error = function(e) impute_mean(X_tr_miss, X_te_full))
  si_rfi  <- tryCatch(impute_rfi(X_tr_miss, X_te_full),
                      error = function(e) impute_mean(X_tr_miss, X_te_full))
  for (cl in CLF_NAMES) {
    rows[[length(rows)+1]] <- rec("MeanSI", cl,
                                  fit_clf(cl, si_mean$train, y_tr, si_mean$test))
    rows[[length(rows)+1]] <- rec("KNN",    cl,
                                  fit_clf(cl, si_knn$train,  y_tr, si_knn$test))
    rows[[length(rows)+1]] <- rec("RFI",    cl,
                                  fit_clf(cl, si_rfi$train,  y_tr, si_rfi$test))
  }
  
  cca_keep <- stats::complete.cases(X_tr_miss)
  X_tr_cca <- X_tr_miss[cca_keep, , drop = FALSE]
  y_tr_cca <- y_tr[cca_keep]
  cca_ok   <- nrow(X_tr_cca) >= 10 && length(unique(y_tr_cca)) == 2 &&
    all(table(y_tr_cca) >= 2)
  for (cl in CLF_NAMES) {
    p_cca_te <- if (cca_ok) {
      tryCatch(fit_clf(cl, X_tr_cca, y_tr_cca, X_te_full),
               error = function(e) rep(0.5, nrow(X_te_full)))
    } else rep(0.5, nrow(X_te_full))
    rows[[length(rows)+1]] <- rec("CCA", cl, p_cca_te)
  }
  
  .t0_se <- Sys.time()
  imp_tr_se <- mice_impute(X_tr_miss, M_IMP_SE, y_tr, split_seed + 1L, MAXIT_MICE)
  imp_te_se <- lapply(seq_len(M_IMP_SE), function(i) X_te_full)
  l0 <- tryCatch(mice_se_l0(imp_tr_se, y_tr, imp_te_se), error = function(e) NULL)
  for (meta_cl in CLF_NAMES) {
    p_se_te <- if (!is.null(l0)) {
      tryCatch(run_mice_se(l0$L0_tr, l0$L0_te, y_tr, meta_cl),
               error = function(e) rowMeans(l0$L0_te, na.rm = TRUE))
    } else {
      rowMeans(do.call(cbind, lapply(seq_len(M_IMP_SE), function(k)
        fit_clf(meta_cl, imp_tr_se[[k]], y_tr, X_te_full))), na.rm = TRUE)
    }
    rows[[length(rows)+1]] <- rec("MIE-SE", meta_cl, p_se_te)
  }
  add_timing("MIE-SE", as.numeric(difftime(Sys.time(), .t0_se, units = "secs")))
  
  .t0_impscore <- Sys.time()
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
        for (mi in seq_along(MICE_STACK_METHODS)) {
          meth <- MICE_STACK_METHODS[mi]
          if (meth != "pmm") next
          rows[[length(rows)+1]] <- rec(MICE_METHOD_LABEL[[meth]], cl, sc$test[, mi])
        }
      } else {
        for (meth in MICE_STACK_METHODS) {
          if (meth != "pmm") next
          imp_draws  <- mice_stack_draws[[meth]]
          test_probs <- if (!is.null(imp_draws)) {
            tryCatch(
              rowMeans(do.call(cbind, lapply(imp_draws, function(draw)
                fit_clf(cl, draw, y_tr, X_te_full))), na.rm = TRUE),
              error = function(e) rep(0.5, nrow(X_te_full)))
          } else rep(0.5, nrow(X_te_full))
          rows[[length(rows)+1]] <- rec(MICE_METHOD_LABEL[[meth]], cl, test_probs)
        }
      }
    }
  }
  add_timing("MICE_Stack_ImputeScore",
             as.numeric(difftime(Sys.time(), .t0_impscore, units = "secs")))
  
  if (length(stack_scores_by_clf) > 0) {
    .t0_glmm <- Sys.time()
    fallback_list <- lapply(stack_scores_by_clf, function(sc) rowMeans(sc$test, na.rm = TRUE))
    glmm_stack_data <- tryCatch(
      build_pooled_glmm_stack_data(stack_scores_by_clf, y_tr),
      error = function(e) NULL)
    glmm_result <- tryCatch(
      run_mice_glmm_stack_pooled(glmm_stack_data, fallback_list),
      error = function(e) list(
        pred = rowMeans(do.call(cbind, fallback_list), na.rm = TRUE),
        varcomp = NULL, formula_used = NA_integer_))
    rows[[length(rows)+1]] <- rec("MICE-GLMM-Stack", "All", glmm_result$pred)
    if (!is.null(glmm_result$varcomp) && nrow(glmm_result$varcomp) > 0) {
      varcomp_local[[length(varcomp_local) + 1]] <- data.frame(
        FormulaRung = glmm_result$formula_used,
        glmm_result$varcomp, stringsAsFactors = FALSE)
    }
    add_timing("MICE_GLMM_Stack_MetaFit",
               as.numeric(difftime(Sys.time(), .t0_glmm, units = "secs")))
  }
  
  list(scores  = do.call(rbind, rows),
       timing  = do.call(rbind, timing_local),
       varcomp = do.call(rbind, varcomp_local))
}
# --- 13. Main Experiment Loop -------------------------------------------------
cat("\n=== Imputation/Stacking Benchmark -- 15 Numeric-Predictor Binary Clinical Datasets ===\n")
cat("MICE: PMM only (nominal/ordinal/binary-coded categoricals excluded)\n")
cat(sprintf("Classifiers: %s | Splits: %d | M_IMP: %d | M_IMP_SE: %d\n",
            paste(CLF_NAMES, collapse="/"), N_SPLITS, M_IMP, M_IMP_SE))
cat("MIE-SE: separate result per meta-clf (J48/NB/PART/SMO/RF)\n")
cat(sprintf("MICE-GLMM-Stack: 3-random-effect GLMM (classifier, MICE method, draw-in-method) over %d methods x %d draws x %s classifiers = %d streams, reported once as Classifier=All\n\n",
            length(MICE_STACK_METHODS), M_IMP_STACK, paste(CLF_NAMES, collapse = "/"),
            length(MICE_STACK_METHODS) * M_IMP_STACK * length(CLF_NAMES)))
all_results <- list()
all_timings <- list()
all_varcomp <- list()
for (ds in DATASET_REGISTRY) {
  cat(sprintf(">>> Loading: %s\n", ds$name))
  DATASET <- tryCatch(ds$loader(), error = function(e) {
    message(sprintf("  [SKIP] %s: %s", ds$name, conditionMessage(e))); NULL
  })
  if (is.null(DATASET) || nrow(DATASET) < 80) {
    cat(sprintf("  [SKIP] %s -- n=%s\n", ds$name,
                if (is.null(DATASET)) "NA" else nrow(DATASET))); next
  }
  PRED_NAMES <- setdiff(names(DATASET), "y")
  non_num <- PRED_NAMES[!sapply(DATASET[, PRED_NAMES, drop=FALSE], is.numeric)]
  if (length(non_num) > 0) {
    cat(sprintf("  [SKIP] %s -- non-numeric predictors: %s\n",
                ds$name, paste(non_num, collapse=", "))); next
  }
  cat(sprintf("  n=%d  p=%d  pos=%d  neg=%d\n",
              nrow(DATASET), length(PRED_NAMES),
              sum(DATASET$y == "pos"), sum(DATASET$y == "neg")))
  
  for (mech in MISS_MECHS) {
    cat(sprintf("  [%s] %s: ", ds$name, mech))
    for (miss_rate in MISS_RATES) {
      cat(sprintf("%g%%", miss_rate * 100))
      for (sp in seq_len(N_SPLITS)) {
        sp_seed <- sp * 137L + as.integer(miss_rate * 1000) + nchar(ds$name)
        set.seed(sp_seed)
        tr_idx <- caret::createDataPartition(DATASET$y, p = TRAIN_FRAC,
                                             list = FALSE)[, 1]
        te_idx <- setdiff(seq_len(nrow(DATASET)), tr_idx)
        X_tr <- DATASET[tr_idx, PRED_NAMES, drop = FALSE]
        y_tr <- DATASET[tr_idx, "y"]
        X_te <- DATASET[te_idx, PRED_NAMES, drop = FALSE]
        y_te <- DATASET[te_idx, "y"]
        
        base <- do.call(rbind, lapply(CLF_NAMES, function(cl) {
          p_te <- fit_clf(cl, X_tr, y_tr, X_te)
          mt   <- compute_metrics(p_te, y_te)
          data.frame(Method = "Complete", Classifier = cl, t(mt),
                     stringsAsFactors = FALSE)
        }))
        base$Dataset      <- ds$name; base$MissMech     <- mech
        base$MissRate     <- 0;       base$Split         <- sp
        base$ActualMiss   <- 0
        all_results[[length(all_results)+1]] <- base
        
        X_tr_miss_check <- tryCatch(
          induce_missing(X_tr, miss_rate, mech, seed = sp_seed), error = function(e) X_tr)
        actual_miss_pct <- round(mean(is.na(X_tr_miss_check)) * 100, 2)
        if (sp == 1)
          cat(sprintf(" [actual=%.1f%%]", actual_miss_pct))
        
        res_pair <- tryCatch(
          run_one(mech, miss_rate, sp_seed, X_tr, y_tr, X_te, y_te),
          error = function(e) {
            message(sprintf("\n  [ERR %s sp=%d mr=%.0f%%] %s",
                            ds$name, sp, miss_rate*100, conditionMessage(e)))
            NULL
          })
        if (!is.null(res_pair)) {
          res <- res_pair$scores
          res$Dataset    <- ds$name;   res$MissMech  <- mech
          res$MissRate   <- miss_rate; res$Split      <- sp
          res$ActualMiss <- actual_miss_pct / 100
          all_results[[length(all_results)+1]] <- res
          
          tmg <- res_pair$timing
          if (!is.null(tmg) && nrow(tmg) > 0) {
            tmg$Dataset  <- ds$name;   tmg$MissMech <- mech
            tmg$MissRate <- miss_rate; tmg$Split     <- sp
            all_timings[[length(all_timings)+1]] <- tmg
          }
          
          vc <- res_pair$varcomp
          if (!is.null(vc) && nrow(vc) > 0) {
            vc$Dataset  <- ds$name;   vc$MissMech <- mech
            vc$MissRate <- miss_rate; vc$Split     <- sp
            all_varcomp[[length(all_varcomp)+1]] <- vc
          }
        }
        cat(".")
      }
      cat(" ")
    }
    cat("done\n")
  }
}
cat("\n=== All runs complete ===\n")
# --- 14. Combine & Save ------------------------------------------------------
combined           <- do.call(rbind, all_results)
rownames(combined) <- NULL
combined$MissRate  <- as.numeric(combined$MissRate)
combined$MissMech  <- factor(combined$MissMech, levels = MISS_MECHS)
combined$Method    <- factor(combined$Method,   levels = METHOD_ORDER)
write.csv(combined, file.path(OUTPUT_DIR, "Multi_HAS_results.csv"), row.names = FALSE)
cat("Saved: Multi_HAS_results.csv\n")
# --- 14a. Actual Missingness Summary -----------------------------------------
actual_miss_summary <- combined %>%
  dplyr::filter(MissRate > 0) %>%
  dplyr::distinct(Dataset, MissMech, MissRate, Split, ActualMiss) %>%
  dplyr::group_by(Dataset, MissMech, MissRate) %>%
  dplyr::summarise(
    N                = dplyr::n(),
    NominalMiss_pct  = unique(MissRate) * 100,
    ActualMiss_mean  = round(mean(ActualMiss, na.rm = TRUE) * 100, 2),
    ActualMiss_sd     = round(sd(ActualMiss,   na.rm = TRUE) * 100, 2),
    ActualMiss_min    = round(min(ActualMiss,  na.rm = TRUE) * 100, 2),
    ActualMiss_max    = round(max(ActualMiss,  na.rm = TRUE) * 100, 2),
    .groups = "drop") %>%
  dplyr::arrange(Dataset, MissMech, MissRate)
write.csv(actual_miss_summary,
          file.path(OUTPUT_DIR, "Multi_ActualMissingness_Summary.csv"),
          row.names = FALSE)
cat("Saved: Multi_ActualMissingness_Summary.csv\n")
# --- 14b. Computational Cost / Runtime Summary -------------------------------
timing_combined <- do.call(rbind, all_timings)
if (!is.null(timing_combined) && nrow(timing_combined) > 0) {
  rownames(timing_combined) <- NULL
  write.csv(timing_combined, file.path(OUTPUT_DIR, "Multi_Runtime_Raw.csv"),
            row.names = FALSE)
  
  wide_t <- timing_combined %>%
    tidyr::pivot_wider(names_from = Method, values_from = RunTime_sec)
  if (all(c("MICE_Stack_ImputeScore", "MICE_GLMM_Stack_MetaFit") %in% names(wide_t))) {
    total_rows <- wide_t %>%
      dplyr::transmute(
        Dataset, MissMech, MissRate, Split,
        Method = "MICE_GLMM_Stack_Total",
        RunTime_sec = MICE_Stack_ImputeScore + MICE_GLMM_Stack_MetaFit)
    timing_combined <- dplyr::bind_rows(timing_combined, total_rows)
  }
  
  runtime_summary <- timing_combined %>%
    dplyr::group_by(MissMech, Method) %>%
    dplyr::summarise(
      N          = dplyr::n(),
      Mean_sec   = round(mean(RunTime_sec, na.rm = TRUE), 4),
      SD_sec     = round(sd(RunTime_sec,   na.rm = TRUE), 4),
      Min_sec    = round(min(RunTime_sec,  na.rm = TRUE), 4),
      Max_sec    = round(max(RunTime_sec,  na.rm = TRUE), 4),
      .groups = "drop")
  write.csv(runtime_summary, file.path(OUTPUT_DIR, "Multi_Runtime_Summary.csv"),
            row.names = FALSE)
  cat("Saved: Multi_Runtime_Raw.csv, Multi_Runtime_Summary.csv\n")
  
  runtime_tex_label <- c(
    `MIE-SE`               = "MIE-SE (Stacking)",
    MICE_Stack_ImputeScore  = "MICE-GLMM-Stack -- Imputation + Scoring (shared)",
    MICE_GLMM_Stack_MetaFit = "MICE-GLMM-Stack -- Meta-GLMM Fit (incremental)",
    MICE_GLMM_Stack_Total   = "\\textbf{MICE-GLMM-Stack -- Total}")
  runtime_order <- c("MIE-SE", "MICE_Stack_ImputeScore",
                     "MICE_GLMM_Stack_MetaFit", "MICE_GLMM_Stack_Total")
  
  make_runtime_latex_table <- function(rs, mech) {
    rs <- rs[rs$MissMech == mech, , drop = FALSE]
    missing_methods <- setdiff(runtime_order, rs$Method)
    if (length(missing_methods) > 0)
      warning(sprintf(
        "Runtime table (%s) is missing expected row(s): %s -- the LaTeX table will be incomplete.",
        mech, paste(missing_methods, collapse = ", ")), call. = FALSE)
    
    present_order <- intersect(runtime_order, rs$Method)
    rs <- rs[match(present_order, rs$Method), ]
    rs <- rs[!is.na(rs$Method), , drop = FALSE]
    if (nrow(rs) == 0)
      warning(sprintf(
        "Runtime summary (%s) had no rows matching runtime_order -- Multi_Runtime_Table_%s.tex will be an empty table.",
        mech, mech), call. = FALSE)
    cap <- paste0(
      "Computational cost comparison between the paper-style MIE-SE ",
      "stacking ensemble and the proposed MICE-GLMM-Stack extension, under ",
      sprintf("the %s missingness mechanism. ", mech),
      "Wall-clock time (seconds) is measured per train--test split, ",
      "averaged ($\\text{mean}\\pm\\text{SD}$) over all datasets, missingness ",
      "rates, and splits ($N$ runs per row). ",
      "\\textit{Imputation + Scoring} is the shared cost of generating the ",
      "MICE base-method imputations and per-classifier out-of-fold scores ",
      "(also reused by the single MICE-PMM reporting row); ",
      "\\textit{Meta-GLMM Fit} is the incremental cost of fitting and ",
      "predicting the 3-random-effect pooled GLMM on top of that shared ",
      "step; \\textit{Total} is their sum and is the fair, like-for-like ",
      "figure to compare directly against MIE-SE.")
    lines <- c(
      "\\begin{table}[!ht]",
      "  \\centering",
      sprintf("  \\caption{%s}", cap),
      sprintf("  \\label{tab:runtime_comparison_%s}", tolower(mech)),
      "  \\begin{tabular}{@{}lrrr@{}}",
      "  \\toprule",
      "  \\textbf{Method / Component} & \\textbf{N} & \\textbf{Mean $\\pm$ SD (s)} & \\textbf{Range (s)} \\\\",
      "  \\midrule")
    for (i in seq_len(nrow(rs))) {
      lbl <- runtime_tex_label[[rs$Method[i]]]
      lines <- c(lines, sprintf(
        "  %s & %d & %.3f $\\pm$ %.3f & [%.3f, %.3f] \\\\",
        lbl, rs$N[i], rs$Mean_sec[i], rs$SD_sec[i], rs$Min_sec[i], rs$Max_sec[i]))
      if (rs$Method[i] == "MICE_GLMM_Stack_MetaFit")
        lines <- c(lines, "  \\midrule")
    }
    lines <- c(lines, "  \\bottomrule", "  \\end{tabular}", "\\end{table}", "")
    paste(lines, collapse = "\n")
  }
  
  for (mech in MISS_MECHS) {
    runtime_tex <- make_runtime_latex_table(runtime_summary, mech)
    out_file <- sprintf("Multi_Runtime_Table_%s.tex", mech)
    writeLines(runtime_tex, file.path(OUTPUT_DIR, out_file))
    cat(sprintf("Saved: %s\n", out_file))
  }
} else {
  cat("[WARN] No timing data captured -- skipping runtime summary/table.\n")
}
# --- 14c. Variance-Component Summary ------------------------------------------
varcomp_combined <- do.call(rbind, all_varcomp)
if (!is.null(varcomp_combined) && nrow(varcomp_combined) > 0) {
  rownames(varcomp_combined) <- NULL
  write.csv(varcomp_combined,
            file.path(OUTPUT_DIR, "Multi_VarianceComponents_Raw.csv"),
            row.names = FALSE)
  
  varcomp_combined <- varcomp_combined %>%
    dplyr::mutate(
      Component = dplyr::case_when(
        grp == "clf_id" & var1 == "(Intercept)" & is.na(var2) ~ "Classifier -- intercept",
        grp == "clf_id" & var1 == "logit_p"     & is.na(var2) ~ "Classifier -- slope (logit p)",
        grp == "clf_id" & !is.na(var2)                        ~ "Classifier -- slope/intercept covariance",
        grp %in% c("draw_id:method_id", "method_id:draw_id")  ~ "Method-within-draw (nested) intercept",
        grp == "method_id"                                    ~ "Method intercept (crossed)",
        grp == "draw_id"                                      ~ "Draw intercept (crossed)",
        TRUE ~ paste0(grp, ifelse(is.na(var1), "", paste0(" -- ", var1)))))
  
  varcomp_summary <- varcomp_combined %>%
    dplyr::group_by(MissMech, Component, FormulaRung) %>%
    dplyr::summarise(
      N          = dplyr::n(),
      Mean_vcov  = round(mean(vcov,  na.rm = TRUE), 5),
      SD_vcov    = round(sd(vcov,    na.rm = TRUE), 5),
      Mean_sdcor = round(mean(sdcor, na.rm = TRUE), 5),
      .groups = "drop") %>%
    dplyr::arrange(MissMech, FormulaRung, dplyr::desc(Mean_vcov))
  write.csv(varcomp_summary,
            file.path(OUTPUT_DIR, "Multi_VarianceComponents_Summary.csv"),
            row.names = FALSE)
  cat("Saved: Multi_VarianceComponents_Raw.csv, Multi_VarianceComponents_Summary.csv\n")
  
  rung_share <- varcomp_combined %>%
    dplyr::distinct(Dataset, MissMech, MissRate, Split, FormulaRung) %>%
    dplyr::count(MissMech, FormulaRung, name = "N_splits") %>%
    dplyr::group_by(MissMech) %>%
    dplyr::mutate(Share = round(N_splits / sum(N_splits), 4)) %>%
    dplyr::ungroup()
  write.csv(rung_share,
            file.path(OUTPUT_DIR, "Multi_VarianceComponents_RungShare.csv"),
            row.names = FALSE)
  cat("Saved: Multi_VarianceComponents_RungShare.csv\n")
  
  make_varcomp_latex_table <- function(vs, mech) {
    vs1 <- vs %>% dplyr::filter(MissMech == mech, FormulaRung == 1)
    if (nrow(vs1) == 0)
      vs1 <- vs %>% dplyr::filter(MissMech == mech,
                                  FormulaRung == suppressWarnings(min(FormulaRung[MissMech == mech], na.rm = TRUE)))
    cap <- paste0(
      "Empirical variance-component estimates from the pooled MICE-GLMM-Stack ",
      "model (Equation~\\eqref{eq:glmm}), averaged ($\\text{mean}\\pm\\text{SD}$) ",
      sprintf("over every train--test split under the %s missingness mechanism ", mech),
      "for which the full model (rung~1 of the ",
      "fallback sequence, Section~\\ref{sec:glmm}) converged. ",
      "$\\widehat{\\mathrm{sdcor}}$ is the estimated standard deviation (or, for ",
      "the slope--intercept term, correlation) on the link scale; $N$ is the ",
      "number of splits contributing to the row.")
    lines <- c(
      "\\begin{table}[!ht]",
      "  \\centering",
      sprintf("  \\caption{%s}", cap),
      sprintf("  \\label{tab:variance_components_%s}", tolower(mech)),
      "  \\begin{tabular}{@{}lrrr@{}}",
      "  \\toprule",
      "  \\textbf{Random-effects component} & \\textbf{N} & \\textbf{Mean $\\pm$ SD (variance)} & \\textbf{Mean SD/cor} \\\\",
      "  \\midrule")
    for (i in seq_len(nrow(vs1))) {
      lines <- c(lines, sprintf(
        "  %s & %d & %.4f $\\pm$ %.4f & %.4f \\\\",
        vs1$Component[i], vs1$N[i], vs1$Mean_vcov[i], vs1$SD_vcov[i], vs1$Mean_sdcor[i]))
    }
    lines <- c(lines, "  \\bottomrule", "  \\end{tabular}", "\\end{table}", "")
    paste(lines, collapse = "\n")
  }
  for (mech in MISS_MECHS) {
    varcomp_tex <- make_varcomp_latex_table(varcomp_summary, mech)
    out_file <- sprintf("Multi_VarianceComponents_Table_%s.tex", mech)
    writeLines(varcomp_tex, file.path(OUTPUT_DIR, out_file))
    cat(sprintf("Saved: %s\n", out_file))
  }
} else {
  cat("[WARN] No variance-component data captured -- skipping variance-component summary/table.\n")
}
# --- 15. Summaries ------------------------------------------------------------
per_ds_summ <- combined %>%
  dplyr::group_by(Dataset, MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(
    AUC_mean   = round(mean(AUC,   na.rm=TRUE), 4),
    AUC_sd     = round(sd(AUC,     na.rm=TRUE), 4),
    Brier_mean = round(mean(Brier, na.rm=TRUE), 4),
    Brier_sd   = round(sd(Brier,   na.rm=TRUE), 4),
    .groups = "drop")
write.csv(per_ds_summ,
          file.path(OUTPUT_DIR, "Multi_HAS_per_dataset_summary.csv"), row.names = FALSE)
grand_summ <- combined %>%
  dplyr::group_by(MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(
    AUC_mean   = round(mean(AUC,   na.rm=TRUE), 4),
    AUC_sd     = round(sd(AUC,     na.rm=TRUE), 4),
    Brier_mean = round(mean(Brier, na.rm=TRUE), 4),
    Brier_sd   = round(sd(Brier,   na.rm=TRUE), 4),
    .groups = "drop")
write.csv(grand_summ,
          file.path(OUTPUT_DIR, "Multi_HAS_grand_summary.csv"), row.names = FALSE)
cat("Saved: summaries\n")
# --- 16. Wilcoxon Signed-Rank Tests -------------------------------------------
# NOTE: Updated to the FULL 30-comparison family (6 baselines x 5 classifiers):
# MeanSI, KNN, RFI, CCA, MICE-PMM, and MIE-SE, each crossed with J48/NB/PART/SMO/RF.
# This matches the design reported throughout the paper (Section 2.8), where
# CCA and MICE-PMM are included in the formal Wilcoxon family for completeness,
# notwithstanding the two caveats noted there (CCA discards the missingness
# mechanism entirely; MICE-PMM is one of the four MICE base methods pooled
# inside MICE-GLMM-Stack itself).
INNER_FAMILY_METHODS <- c("MeanSI", "KNN", "RFI", "CCA", "MICE-PMM", "MIE-SE")
STACK_METHODS <- c("MICE-GLMM-Stack")
STACK_CLF     <- "All"
ds_level_wt <- combined %>%
  dplyr::group_by(Dataset, MissMech, MissRate, Method, Classifier) %>%
  dplyr::summarise(AUC_ds   = mean(AUC,   na.rm = TRUE),
                   Brier_ds = mean(Brier, na.rm = TRUE),
                   .groups  = "drop")
n_comp <- length(INNER_FAMILY_METHODS) * length(CLF_NAMES)  # Bonferroni: 30
compute_stack_wilcoxon <- function(stack_method) {
  rows_local <- list()
  stack_ds <- ds_level_wt %>%
    dplyr::filter(Method == stack_method, Classifier == STACK_CLF) %>%
    dplyr::select(Dataset, MissMech, MissRate,
                  AUC_stack = AUC_ds, Brier_stack = Brier_ds)
  if (nrow(stack_ds) == 0) return(NULL)
  
  for (meth in INNER_FAMILY_METHODS) {
    for (cl in CLF_NAMES) {
      comp_ds <- ds_level_wt %>%
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
            n  <- sum(!is.na(diff_vec))
            if (n < 2) next
            wt <- tryCatch(
              wilcox.test(stack_col, comp_col, paired = TRUE, exact = FALSE),
              error = function(e) NULL)
            if (is.null(wt)) next
            p_adj  <- round(min(wt$p.value * n_comp, 1), 5)
            better <- if (metric == "AUC") diff_vec > 0 else diff_vec < 0
            worse  <- if (metric == "AUC") diff_vec < 0 else diff_vec > 0
            rows_local[[length(rows_local) + 1]] <- data.frame(
              StackMethod        = stack_method,
              Metric             = metric,
              MissMech           = mech,
              MissRate           = mr,
              MissRate_pct       = paste0(round(mr * 100), "%"),
              Competitor         = meth,
              CompetitorClf      = cl,
              n_pairs            = n,
              W                  = round(wt$statistic, 1),
              p_value            = round(wt$p.value, 5),
              p_adj_Bonf         = p_adj,
              Signif             = dplyr::case_when(p_adj < 0.001 ~ "***",
                                                    p_adj < 0.01  ~ "**",
                                                    p_adj < 0.05  ~ "*",
                                                    TRUE          ~ "ns"),
              mean_diff          = round(mean(diff_vec, na.rm = TRUE), 4),
              sd_diff            = round(sd(diff_vec,   na.rm = TRUE), 4),
              Stack_wins         = sum(better,         na.rm = TRUE),
              Ties               = sum(diff_vec == 0,  na.rm = TRUE),
              Stack_losses       = sum(worse,          na.rm = TRUE),
              stringsAsFactors = FALSE)
          }
        }
      }
    }
  }
  if (length(rows_local) == 0) return(NULL)
  do.call(rbind, rows_local)
}
wilcox_df <- do.call(rbind, lapply(STACK_METHODS, compute_stack_wilcoxon))
rownames(wilcox_df) <- NULL
write.csv(wilcox_df, file.path(OUTPUT_DIR, "Multi_HAS_wilcoxon.csv"), row.names = FALSE)
cat("Saved: Multi_HAS_wilcoxon.csv\n")
cat("\n=== Wilcoxon Summary (AUC, 30%) ===\n")
print(wilcox_df[wilcox_df$Metric == "AUC" & abs(wilcox_df$MissRate - 0.30) < 1e-6, ],
      row.names = FALSE)
# --- 16a. LaTeX Helper Functions ---------------------------------------------
method_tex <- function(meth)
  switch(as.character(meth),
         `MICE-GLMM-Stack` = "\\textbf{MICE-GLMM-Stack}",
         `MIE-SE`         = "MIE-SE (Stacking)",
         `MICE-PMM`        = "MICE-PMM",
         CCA               = "CCA",
         MeanSI            = "Mean SI",
         Complete          = "Complete",
         as.character(meth))
fmt_cell <- function(mn, sd, bold = FALSE) {
  if (is.na(mn)) return("{--}")
  sd <- if (is.na(sd)) 0 else sd
  if (bold) sprintf("$\\mathbf{%.3f}{\\pm}\\mathbf{%.3f}$", mn, sd)
  else      sprintf("$%.3f{\\pm}%.3f$", mn, sd)
}
pivot_wide_ms <- function(df, metric) {
  mc  <- paste0(metric, "_mean"); sc <- paste0(metric, "_sd")
  df2 <- df %>%
    dplyr::mutate(col_key = dplyr::case_when(
      abs(MissRate) < 1e-6 ~ "COMP",
      TRUE ~ paste0(as.character(MissMech),
                    sprintf("%02d", as.integer(round(MissRate * 100)))))) %>%
    dplyr::group_by(Method, Classifier, col_key) %>%
    dplyr::summarise(!!mc := mean(.data[[mc]], na.rm=TRUE),
                     !!sc := mean(.data[[sc]], na.rm=TRUE), .groups="drop")
  wm <- df2 %>%
    dplyr::select(Method, Classifier, col_key, val=dplyr::all_of(mc)) %>%
    tidyr::pivot_wider(names_from=col_key, values_from=val)
  ws <- df2 %>%
    dplyr::select(Method, Classifier, col_key, val=dplyr::all_of(sc)) %>%
    tidyr::pivot_wider(names_from=col_key, values_from=val, names_prefix="SD_")
  dplyr::left_join(wm, ws, by=c("Method","Classifier"))
}
`%.%` <- function(a, b) paste0(a, b)
build_latex_table <- function(wide, metric, caption_str, label_str) {
  hb         <- (metric != "Brier")
  mech_key_list <- lapply(MISS_MECHS, function(m)
    paste0(m, sprintf("%02d", as.integer(MISS_RATES * 100))))
  names(mech_key_list) <- MISS_MECHS
  all_mech_keys <- unlist(mech_key_list, use.names = FALSE)
  present    <- intersect(c("COMP", all_mech_keys), names(wide))
  nc         <- sum("COMP" %in% present)
  n_per_mech <- sapply(mech_key_list, function(ks) sum(ks %in% present))
  nd <- nc + sum(n_per_mech)
  wide <- wide %>%
    dplyr::mutate(Method = factor(Method, levels = METHOD_ORDER)) %>%
    dplyr::arrange(Method, Classifier)
  best_row <- sapply(present, function(k) {
    vals <- unlist(wide[[k]]); if (all(is.na(vals))) return(NA_integer_)
    if (hb) which(vals == max(vals, na.rm=TRUE))[1]
    else    which(vals == min(vals, na.rm=TRUE))[1]
  })
  lines <- c("\\begin{table}[!ht]","  \\centering",
             sprintf("  \\caption{%s}", caption_str),
             sprintf("  \\label{%s}", label_str),
             "  \\resizebox{\\linewidth}{!}{%",
             sprintf("  \\begin{tabular}{@{}ll%s@{}}", strrep("c", nd)),
             "  \\toprule")
  sc <- c("\\textbf{Method}","\\textbf{Clf}")
  if (nc > 0) sc <- c(sc, sprintf("\\multicolumn{%d}{c}{\\textbf{Complete}}", nc))
  for (m in MISS_MECHS)
    if (n_per_mech[[m]] > 0)
      sc <- c(sc, sprintf("\\multicolumn{%d}{c}{\\textbf{%s}}", n_per_mech[[m]], m))
  lines <- c(lines, sprintf("  %s \\\\", paste(sc, collapse=" & ")))
  cs2 <- 3L; cmi <- character(0)
  if (nc > 0) { cmi <- c(cmi, sprintf("\\cmidrule(lr){%d-%d}", cs2, cs2+nc-1)); cs2 <- cs2+nc }
  for (m in MISS_MECHS)
    if (n_per_mech[[m]] > 0) {
      cmi <- c(cmi, sprintf("\\cmidrule(lr){%d-%d}", cs2, cs2+n_per_mech[[m]]-1))
      cs2 <- cs2 + n_per_mech[[m]]
    }
  if (length(cmi) > 0) lines <- c(lines, paste("  ", paste(cmi, collapse=" ")))
  rl   <- paste0(as.integer(MISS_RATES * 100), "\\%")
  sub  <- c("","")
  if (nc > 0) sub <- c(sub, "--")
  for (m in MISS_MECHS)
    if (n_per_mech[[m]] > 0) sub <- c(sub, rl[seq_len(n_per_mech[[m]])])
  lines <- c(lines, sprintf("  %s \\\\", paste(sub, collapse=" & ")), "  \\midrule")
  prev <- ""
  for (i in seq_len(nrow(wide))) {
    mth <- as.character(wide$Method[i])
    if (mth != prev && i > 1) lines <- c(lines, "  \\midrule")
    prev <- mth
    cells <- sapply(seq_along(present), function(ci) {
      k  <- present[ci]
      mn <- unlist(wide[[k]])[i]; sd <- unlist(wide[[paste0("SD_",k)]])[i]
      fmt_cell(mn, sd, bold = (!is.na(best_row[ci]) && best_row[ci] == i))
    })
    lines <- c(lines, sprintf("  %s & %s & %s \\\\",
                              method_tex(mth), wide$Classifier[i],
                              paste(cells, collapse=" & ")))
  }
  lines <- c(lines,"  \\bottomrule","  \\end{tabular}%","  }","\\end{table}","")
  paste(lines, collapse="\n")
}
make_grand_table <- function(gs, metric="AUC") {
  note <- if (metric != "Brier") "higher is better" else "lower is better"
  primary_note <- if (metric == "AUC")
    "Primary metric -- threshold-independent discrimination. " else
      if (metric == "Brier")
        "Secondary metric -- probability calibration (lower is better). "
  else ""
  cap  <- sprintf(
    "Comprehensive comparison of imputation and classification methods: mean %s " %.%
      "($\\text{mean}\\pm\\text{SD}$) averaged over 15 numeric-predictor binary clinical " %.%
      "benchmark datasets and %d stratified train--test splits (70/30; %s). " %.%
      "%sMIE-SE uses PMM exclusively; MICE-PMM reports the predictive-mean-matching " %.%
      "MICE base method on its own. \\textit{CCA}: complete case analysis (listwise " %.%
      "deletion of training rows with any missing predictor cell, no imputation). " %.%
      "\\textit{Complete}: complete-data reference with no induced missingness. " %.%
      "\\textit{MICE-GLMM-Stack}: pooled 3-random-effect GLMM meta-learner spanning all " %.%
      "classifiers and all 4 MICE base methods (pmm/cart/norm/rf) internally, aggregated " %.%
      "jointly; only MICE-PMM among these 4 base methods is also reported as its own " %.%
      "standalone competitor row. " %.%
      "Boldface denotes the best-performing method per column. " %.%
      "Missing data were induced independently under each of: %s.",
    metric, N_SPLITS, note, primary_note,
    paste(sapply(MISS_MECHS, function(m) unname(mech_labels[m])), collapse = "; "))
  build_latex_table(pivot_wide_ms(gs, metric), metric, cap,
                    sprintf("tab:grand_%s", tolower(metric)))
}
make_per_dataset_table <- function(ds_name, metric="AUC") {
  note <- if (metric != "Brier") "higher is better" else "lower is better"
  cap  <- sprintf(
    "%s: mean %s ($\\text{mean}\\pm\\text{SD}$) over %d stratified train--test splits " %.%
      "(70/30; %s). Boldface indicates the best-performing method per column.",
    gsub("_"," ",ds_name), metric, N_SPLITS, note)
  df_ds <- per_ds_summ %>% dplyr::filter(Dataset == ds_name)
  if (nrow(df_ds) == 0) return("")
  build_latex_table(pivot_wide_ms(df_ds, metric), metric, cap,
                    sprintf("tab:%s_%s", gsub("[^A-Za-z0-9]","",ds_name), tolower(metric)))
}
latex_preamble <- function(fs=10)
  c(sprintf("\\documentclass[%dpt]{article}", fs),
    "\\usepackage[utf8]{inputenc}","\\usepackage[T1]{fontenc}",
    "\\usepackage{booktabs,geometry,amsmath,graphicx,bm}",
    "\\usepackage[table]{xcolor}",
    "\\geometry{margin=1.2cm,landscape,a3paper}",
    "\\begin{document}","")
# --- 16b. LaTeX Tables: one per mechanism x metric x rate --------------------
make_wilcoxon_latex_rate <- function(wdf, metric_sel, miss_rate, miss_mech = "MCAR",
                                     stack_method = "MICE-GLMM-Stack",
                                     first_ref_label = NULL) {
  df <- wdf[wdf$Metric == metric_sel & wdf$MissMech == miss_mech &
              abs(wdf$MissRate - miss_rate) < 1e-6, ]
  if (nrow(df) == 0) return("")
  rate_pct  <- paste0(round(miss_rate * 100), "\\%")
  lbl       <- sprintf("tab:wilcox_%s_%s_%s_%02d", tolower(stack_method),
                       tolower(miss_mech), tolower(metric_sel), round(miss_rate * 100))
  col_spec  <- "llcccccc"
  stack_tex <- method_tex(stack_method)
  mech_desc <- unname(mech_labels[miss_mech])
  if (is.na(mech_desc) || is.null(mech_desc)) mech_desc <- miss_mech
  
  direction_note <- if (metric_sel == "AUC")
    sprintf("Positive mean diff.\\ = %s wins. $H_1$: AUC(%s) $>$ AUC(Competitor).",
            stack_method, stack_method)
  else
    sprintf("Negative mean diff.\\ = %s wins (lower Brier). $H_1$: Brier(%s) $<$ Brier(Competitor).",
            stack_method, stack_method)
  
  re_desc <- "single 3-random-effect GLMM per split, pooling classifier, MICE method, and imputation draw -- see Sec.9c"
  
  n_ds <- length(unique(combined$Dataset))
  if (is.null(first_ref_label)) {
    cap <- sprintf(
      paste0("Wilcoxon signed-rank test: the pooled \\textbf{%s/All} result ",
             "(%s) versus every imputation ",
             "method $\\times$ classifier competitor, at %s %s (%s). ",
             "Each comparison uses %d paired dataset-level mean %s observations. ",
             "$W$ = Wilcoxon statistic; $p_{\\text{adj}}$ = Bonferroni-corrected $p$-value ",
             "(adjusted for %d comparisons: MeanSI, KNN, RFI, CCA, MICE-PMM, and MIE-SE, ",
             "each $\\times$ 5 classifiers); ",
             "Mean diff.\\ $\\pm$ SD = mean(%s $-$ Competitor) $\\pm$ SD across %d datasets; ",
             "W/L = number of datasets where %s wins~/ loses. ",
             "$^{***}\\,p_{\\text{adj}}<0.001$, $^{**}\\,p_{\\text{adj}}<0.01$, ",
             "$^{*}\\,p_{\\text{adj}}<0.05$, ns~=~not significant."),
      stack_method, re_desc, rate_pct, mech_desc, metric_sel, n_ds, metric_sel, n_comp,
      stack_method, n_ds, stack_method)
  } else {
    cap <- sprintf(
      paste0("Wilcoxon signed-rank test: %s/All versus every imputation ",
             "method $\\times$ classifier competitor, at %s %s (%s). Same design as ",
             "Table~\\ref{%s}."),
      stack_method, rate_pct, mech_desc, metric_sel, first_ref_label)
  }
  
  header_row <- paste(
    c("Competitor", "Clf", "$W$", "$p_{\\text{adj}}$",
      "Signif.", "Mean diff.\\ $\\pm$ SD", "W/L"),
    collapse = " & ")
  
  body_rows <- character(0)
  prev_meth <- ""
  for (meth in INNER_FAMILY_METHODS) {
    for (cl in CLF_NAMES) {
      if (meth != prev_meth && prev_meth != "") body_rows <- c(body_rows, "  \\addlinespace")
      prev_meth <- meth
      r <- df[df$Competitor == meth & df$CompetitorClf == cl, ]
      if (nrow(r) == 0) {
        row_cells <- c(method_tex(meth), cl, "---", "---", "---", "---", "---")
      } else {
        sig      <- r$Signif[1]
        cell_sig <- if (sig != "ns") paste0("$^{", sig, "}$") else "ns"
        mean_sd  <- sprintf("$%.4f \\pm %.4f$", r$mean_diff[1], r$sd_diff[1])
        wl       <- sprintf("%d/%d", r$Stack_wins[1], r$Stack_losses[1])
        row_cells <- c(method_tex(meth), cl,
                       sprintf("%.1f", r$W[1]),
                       sprintf("%.5f", r$p_adj_Bonf[1]),
                       cell_sig, mean_sd, wl)
      }
      body_rows <- c(body_rows,
                     paste0("  ", paste(row_cells, collapse = " & "), " \\\\"))
    }
  }
  
  lines <- c(
    "\\begin{table}[!ht]", "  \\centering",
    sprintf("  \\caption{%s}", cap),
    sprintf("  \\label{%s}", lbl),
    "  \\resizebox{\\linewidth}{!}{%",
    sprintf("  \\begin{tabular}{@{}%s@{}}", col_spec),
    "  \\toprule",
    paste0("  ", header_row, " \\\\"),
    "  \\midrule",
    body_rows,
    "  \\bottomrule",
    sprintf("  \\multicolumn{7}{l}{\\scriptsize{%s}}", direction_note),
    "  \\end{tabular}%", "  }",
    "\\end{table}", "")
  paste(lines, collapse = "\n")
}
for (mech in MISS_MECHS) {
  wilcox_tex_file  <- file.path(OUTPUT_DIR,
                                sprintf("Multi_HAS_wilcoxon_tables_%s.tex", mech))
  wilcox_tex_lines <- latex_preamble(10)
  for (stack_m in STACK_METHODS) {
    wilcox_tex_lines <- c(wilcox_tex_lines,
                          "% ============================================================",
                          sprintf("%% Stack method: %s -- Mechanism: %s", stack_m, mech),
                          "% ============================================================",
                          "")
    for (met in c("AUC", "Brier")) {
      wilcox_tex_lines <- c(wilcox_tex_lines,
                            "% ------------------------------------------------------------",
                            sprintf("%% Metric: %s", met),
                            "% ------------------------------------------------------------",
                            "")
      first_lbl <- NULL
      for (mr in MISS_RATES) {
        tbl <- make_wilcoxon_latex_rate(wilcox_df[wilcox_df$StackMethod == stack_m, ],
                                        met, mr, miss_mech = mech, stack_method = stack_m,
                                        first_ref_label = first_lbl)
        wilcox_tex_lines <- c(wilcox_tex_lines, tbl)
        if (is.null(first_lbl))
          first_lbl <- sprintf("tab:wilcox_%s_%s_%s_%02d", tolower(stack_m),
                               tolower(mech), tolower(met), round(mr * 100))
      }
    }
  }
  wilcox_tex_lines <- c(wilcox_tex_lines, "\\end{document}")
  writeLines(wilcox_tex_lines, wilcox_tex_file)
  cat(sprintf("Saved: %s (10 stack-vs-competitor tables: 5 rates x 2 metrics, 30 rows each)\n",
              wilcox_tex_file))
}
# --- 17. LaTeX Tables: grand and per-dataset output --------------------------
tex_file  <- file.path(OUTPUT_DIR, "Multi_HAS_grand_tables.tex")
tex_lines <- latex_preamble(10)
for (met in c("AUC","Brier"))
  tex_lines <- c(tex_lines, make_grand_table(grand_summ, met))
tex_lines <- c(tex_lines, "\\end{document}")
writeLines(tex_lines, tex_file); cat(sprintf("Saved: %s\n", tex_file))
ds_tex_file  <- file.path(OUTPUT_DIR, "Multi_HAS_per_dataset_tables.tex")
ds_tex_lines <- latex_preamble(9)
for (ds_name in unique(combined$Dataset))
  for (met in c("AUC","Brier"))
    ds_tex_lines <- c(ds_tex_lines, make_per_dataset_table(ds_name, met))
ds_tex_lines <- c(ds_tex_lines, "\\end{document}")
writeLines(ds_tex_lines, ds_tex_file); cat(sprintf("Saved: %s\n", ds_tex_file))
# --- 18. Figures ---------------------------------------------------------------
colour_pal <- c(
  Complete    = "#9E9E9E", MeanSI      = "#795548", KNN         = "#FF9800",
  RFI         = "#8BC34A", CCA         = "#607D8B",
  `MICE-PMM`  = "#9C27B0",
  `MICE-GLMM-Stack` = "#43A047", `MIE-SE` = "#00BCD4")
ltype_pal  <- c(Complete="dotted", MeanSI="solid", KNN="solid",
                RFI="solid", CCA="solid",
                `MICE-PMM`="solid",
                `MICE-GLMM-Stack`="solid", `MIE-SE`="twodash")
lsize_pal  <- c(Complete=0.5, MeanSI=0.5, KNN=0.5, RFI=0.5, CCA=0.5,
                `MICE-PMM`=0.5,
                `MICE-GLMM-Stack`=1.4, `MIE-SE`=0.9)
clf_shape_pal <- setNames(c(16L, 17L, 15L, 18L, 19L), CLF_NAMES)
METHOD_LABELS <- c(
  Complete    = "Complete",          MeanSI      = "Mean (SI)",
  KNN         = "KNN (SI)",          RFI         = "RFI (SI)",
  CCA         = "CCA",
  `MICE-PMM`  = "MICE-PMM",
  `MICE-GLMM-Stack` = "MICE-GLMM-Stack",
  `MIE-SE`   = "MIE-SE")
theme_paper <- function(base = 9)
  theme_bw(base_size = base) +
  theme(
    text              = element_text(family = "serif"),
    panel.grid.minor  = element_blank(),
    panel.grid.major  = element_line(colour = "grey88", linewidth = 0.3),
    panel.border      = element_rect(colour = "grey40", fill = NA, linewidth = 0.5),
    strip.background  = element_rect(fill = "white", colour = "grey40"),
    strip.text        = element_text(face = "bold", size = rel(0.9),
                                     family = "serif"),
    legend.position   = "bottom",
    legend.title      = element_blank(),
    legend.text       = element_text(size = rel(0.85)),
    legend.key.width  = unit(1.6, "lines"),
    plot.title        = element_text(face = "bold", size = rel(1.0),
                                     hjust = 0, family = "serif"),
    plot.caption      = element_text(size = rel(0.75), colour = "grey40",
                                     hjust = 0, face = "italic"),
    axis.title        = element_text(size = rel(0.95), family = "serif"),
    axis.text         = element_text(size = rel(0.85))
  )
clf_raw <- combined %>%
  dplyr::filter(Method %in% METHOD_ORDER, MissMech == "MCAR") %>%
  dplyr::group_by(MissRate, Method, Classifier) %>%
  dplyr::summarise(Mean_AUC   = mean(AUC,   na.rm = TRUE),
                   SD_AUC     = sd(AUC,     na.rm = TRUE),
                   Mean_Brier = mean(Brier, na.rm = TRUE),
                   SD_Brier   = sd(Brier,   na.rm = TRUE),
                   .groups = "drop")
stack_raw <- clf_raw %>% dplyr::filter(Method == "MICE-GLMM-Stack", Classifier == "All")
stack_rep <- purrr::map_dfr(CLF_NAMES, ~ dplyr::mutate(stack_raw, Classifier = .x))
plot_clf <- dplyr::bind_rows(
  clf_raw %>% dplyr::filter(Method != "MICE-GLMM-Stack"),
  stack_rep) %>%
  dplyr::mutate(
    Method      = factor(Method, levels = METHOD_ORDER),
    MethodLabel = factor(dplyr::recode(as.character(Method), !!!METHOD_LABELS),
                         levels = unname(METHOD_LABELS)),
    ClfLabel    = factor(Classifier, levels = CLF_NAMES)) %>%
  dplyr::filter(!is.na(MethodLabel), !is.na(ClfLabel))
mc_levels <- paste0(rep(unname(METHOD_LABELS), each = length(CLF_NAMES)),
                    "/", CLF_NAMES)
mc_levels <- c(mc_levels[!grepl("MICE-GLMM-Stack", mc_levels)],
               "MICE-GLMM-Stack/All")
build_fig56 <- function(mech) {
  clf_ds_raw <- combined %>%
    dplyr::filter(Method %in% METHOD_ORDER, MissMech == mech) %>%
    dplyr::group_by(MissRate, Method, Classifier, Dataset) %>%
    dplyr::summarise(Mean_AUC   = mean(AUC,   na.rm = TRUE),
                     Mean_Brier = mean(Brier, na.rm = TRUE),
                     .groups = "drop")
  
  stack_ds_raw <- clf_ds_raw %>% dplyr::filter(Method == "MICE-GLMM-Stack", Classifier == "All")
  stack_ds_rep <- purrr::map_dfr(CLF_NAMES, ~ dplyr::mutate(stack_ds_raw, Classifier = .x))
  
  pcd <- dplyr::bind_rows(
    clf_ds_raw %>% dplyr::filter(Method != "MICE-GLMM-Stack"),
    stack_ds_rep) %>%
    dplyr::mutate(
      Method      = factor(Method, levels = METHOD_ORDER),
      MethodLabel = factor(dplyr::recode(as.character(Method), !!!METHOD_LABELS),
                           levels = unname(METHOD_LABELS)),
      ClfLabel    = factor(Classifier, levels = CLF_NAMES),
      MethodClf   = paste0(as.character(MethodLabel), "/", Classifier)) %>%
    dplyr::filter(!is.na(MethodLabel), !is.na(ClfLabel))
  
  pcd <- pcd %>%
    dplyr::mutate(
      MethodClf = dplyr::case_when(
        Method == "MICE-GLMM-Stack" ~ "MICE-GLMM-Stack/All",
        TRUE                        ~ paste0(as.character(MethodLabel), "/", Classifier)),
      MethodClf = factor(MethodClf, levels = mc_levels))
  
  mech_desc <- unname(mech_labels[mech])
  if (is.na(mech_desc) || is.null(mech_desc)) mech_desc <- mech
  
  p5 <- ggplot(pcd,
               aes(x = MissRate * 100, y = Mean_AUC,
                   colour = MethodLabel,
                   linetype = MethodLabel,
                   shape    = ClfLabel,
                   group    = interaction(MethodLabel, ClfLabel))) +
    geom_line(linewidth = 0.55) +
    geom_point(size = 1.4) +
    facet_wrap(~ Dataset, nrow = 2, scales = "free_y") +
    scale_colour_manual(  values = setNames(colour_pal,  METHOD_LABELS), name = "Method") +
    scale_linetype_manual(values = setNames(ltype_pal,   METHOD_LABELS), name = "Method") +
    scale_shape_manual(   values = clf_shape_pal,                        name = "Classifier") +
    scale_x_continuous(breaks = MISS_RATES * 100,
                       labels = paste0(MISS_RATES * 100, "%")) +
    labs(title   = sprintf("Dataset-Level AUC Trajectories Under %s: All Method x Classifier Combinations", mech_desc),
         caption = "Colour: imputation method; shape: base classifier.",
         x = "Missingness Rate (%)", y = "Mean AUC per Dataset") +
    theme_paper(base = 8) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7)) +
    guides(colour   = guide_legend(nrow = 2, override.aes = list(linewidth = 1.2)),
           linetype = guide_legend(nrow = 2),
           shape    = guide_legend(nrow = 1))
  
  box_dat <- pcd %>%
    dplyr::filter(Method != "Complete") %>%
    dplyr::mutate(
      MissRate_lbl = factor(paste0(as.integer(MissRate * 100), "% ", mech),
                            levels = paste0(as.integer(MISS_RATES * 100), "% ", mech))) %>%
    dplyr::filter(!is.na(MissRate_lbl))
  
  p6 <- ggplot(box_dat,
               aes(x = MethodClf, y = Mean_AUC, fill = MethodLabel)) +
    geom_boxplot(outlier.size = 0.6, linewidth = 0.35, width = 0.7) +
    facet_wrap(~ MissRate_lbl, nrow = 1) +
    scale_fill_manual(values = setNames(colour_pal, METHOD_LABELS)) +
    labs(title   = sprintf("Distribution of Dataset-Level Mean AUC by Method x Classifier Under %s", mech_desc),
         caption = "Each box: fifteen benchmark datasets. x-axis: Method/Classifier combination.",
         x = NULL, y = "Mean AUC per Dataset") +
    theme_paper() +
    theme(axis.text.x   = element_text(angle = 55, hjust = 1, size = 5.5),
          legend.position = "bottom") +
    guides(fill = guide_legend(nrow = 2))
  
  list(p5 = p5, p6 = p6, plot_clf_ds = pcd)
}
fig56_by_mech <- list()
fig_list <- list()
for (mech in MISS_MECHS) {
  figs <- build_fig56(mech)
  fig56_by_mech[[mech]] <- figs
  fig_list <- c(fig_list, list(
    list(fig = figs$p5, file = sprintf("fig5_per_dataset_auc_%s", mech), w = 20, h = 9),
    list(fig = figs$p6, file = sprintf("fig6_auc_boxplot_%s", mech),     w = 24, h = 7)
  ))
}
plot_clf_ds <- fig56_by_mech[[MISS_MECHS[1]]]$plot_clf_ds
for (fl in fig_list) {
  ggsave(file.path(OUTPUT_DIR, paste0(fl$file, ".png")),
         plot   = fl$fig,
         width  = fl$w, height = fl$h, units = "in",
         dpi    = 900)
  ggsave(file.path(OUTPUT_DIR, paste0(fl$file, ".pdf")),
         plot   = fl$fig,
         width  = fl$w, height = fl$h, units = "in",
         device = cairo_pdf)
  cat(sprintf("Saved: %s (.png 900 dpi + .pdf)\n", fl$file))
}
cat("\n=== Grand AUC @ 30% missingness (mean over 15 datasets, all mechanisms) ===\n")
grand_summ %>%
  dplyr::filter(abs(MissRate - 0.30) < 1e-6 | MissRate == 0) %>%
  dplyr::mutate(Method=factor(Method, levels=METHOD_ORDER)) %>%
  dplyr::arrange(MissMech, Method, Classifier) %>%
  dplyr::select(MissMech, Method, Classifier, AUC_mean, Brier_mean) %>%
  as.data.frame() %>% print(row.names=FALSE)
cat("\n=== DONE ===\n")
# --- 20. Kendall's W Rank Tables ----------------------------------------------
compute_kendall_W <- function(rank_mat) {
  n <- nrow(rank_mat)
  k <- ncol(rank_mat)
  if (n < 2 || k < 2) return(list(W = NA_real_, chi2 = NA_real_,
                                  df = NA_integer_, p_value = NA_real_))
  
  R_i   <- rowSums(rank_mat, na.rm = TRUE)
  R_bar <- k * (n + 1) / 2
  
  SS <- sum((R_i - R_bar)^2)
  
  CF_j <- sapply(seq_len(k), function(j) {
    rj <- rank_mat[, j]
    rj <- rj[!is.na(rj)]
    tbl <- table(rj)
    t_g <- as.numeric(tbl)
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
    dplyr::filter(
      Method != "Complete" &
        abs(MissRate - miss_rate_val) < 1e-6
    )
  
  if (nrow(sub) == 0) {
    message(sprintf("[Kendall] No %s data at %g%%", mech, miss_rate_val * 100))
    return(NULL)
  }
  
  agg <- sub %>%
    dplyr::group_by(Method, Classifier, Dataset) %>%
    dplyr::summarise(metric_mean = mean(.data[[metric]], na.rm = TRUE),
                     .groups = "drop") %>%
    dplyr::mutate(
      Method    = factor(Method, levels = METHOD_ORDER),
      MethodClf = paste(Method, Classifier, sep = "/")
    )
  
  row_order <- c(
    paste("MeanSI",          CLF_NAMES, sep="/"),
    paste("KNN",             CLF_NAMES, sep="/"),
    paste("RFI",             CLF_NAMES, sep="/"),
    paste("CCA",             CLF_NAMES, sep="/"),
    paste("MICE-PMM",        CLF_NAMES, sep="/"),
    "MICE-GLMM-Stack/All",
    paste("MIE-SE",         CLF_NAMES, sep="/")
  )
  row_order <- row_order[row_order %in% unique(agg$MethodClf)]
  
  wide_auc <- agg %>%
    dplyr::select(MethodClf, Dataset, metric_mean) %>%
    tidyr::pivot_wider(names_from = Dataset, values_from = metric_mean) %>%
    dplyr::mutate(MethodClf = factor(MethodClf, levels = row_order)) %>%
    dplyr::arrange(MethodClf)
  
  ds_cols <- setdiff(names(wide_auc), "MethodClf")
  
  sign_flip <- if (higher_is_better) -1 else 1
  rank_wide <- wide_auc
  for (dc in ds_cols) {
    rank_wide[[dc]] <- rank(sign_flip * wide_auc[[dc]],
                            ties.method = "average", na.last = "keep")
  }
  
  rank_mat <- as.matrix(rank_wide[, ds_cols, drop = FALSE])
  rownames(rank_mat) <- as.character(rank_wide$MethodClf)
  
  rank_wide$MeanRank <- round(rowMeans(rank_mat, na.rm = TRUE), 2)
  
  rank_wide$OverallRank <- rank(rank_wide$MeanRank, ties.method = "average")
  
  rank_wide <- rank_wide[order(rank_wide$OverallRank), ]
  
  rank_mat <- as.matrix(rank_wide[, ds_cols, drop = FALSE])
  rownames(rank_mat) <- as.character(rank_wide$MethodClf)
  
  kw <- compute_kendall_W(rank_mat)
  
  list(
    miss_rate   = miss_rate_val,
    metric      = metric,
    mech        = mech,
    auc_wide    = wide_auc,
    rank_wide   = rank_wide,
    rank_mat    = rank_mat,
    datasets    = ds_cols,
    kendall     = kw
  )
}
rank_table_to_latex <- function(rt) {
  mr_pct  <- as.integer(round(rt$miss_rate * 100))
  met     <- rt$metric
  mech    <- if (!is.null(rt$mech)) rt$mech else "MCAR"
  mech_desc <- unname(mech_labels[mech]); if (is.na(mech_desc) || is.null(mech_desc)) mech_desc <- mech
  rw      <- rt$rank_wide
  ds_cols <- rt$datasets
  kw      <- rt$kendall
  n_ds    <- length(ds_cols)
  n_cols  <- n_ds + 2L
  n_rows  <- nrow(rw)
  
  cap <- sprintf(
    paste0("Dataset-level performance rankings of imputation methods ",
           "(\\textbf{%s} metric, %d\\%% %s). ",
           "Rank~1 denotes the best-performing method on each dataset; ",
           "rankings are summarised as the mean rank and an overall rank (sorted ascending) ",
           "across %d binary clinical benchmark datasets. ",
           "Concordance of rankings is assessed by Kendall's coefficient of concordance: ",
           "$W = %.4f$, $\\chi^2(%d) = %.3f$, $p = %s$."),
    met, mr_pct, mech_desc, n_ds,
    ifelse(is.na(kw$W), NA, kw$W),
    kw$df,
    ifelse(is.na(kw$chi2), NA, kw$chi2),
    ifelse(is.na(kw$p_value), "\\text{NA}",
           ifelse(kw$p_value < 0.001, "<0.001",
                  sprintf("%.4f", kw$p_value)))
  )
  lbl <- sprintf("tab:rank_%s_%s%02d", tolower(met), mech, mr_pct)
  
  col_hdrs <- gsub("_", " ", ds_cols)
  
  hdr <- paste(
    c(sprintf("\\textbf{%s}", col_hdrs),
      "\\textbf{Mean Rank}",
      "\\textbf{Overall Rank}"),
    collapse = " & ")
  
  lines <- c(
    "\\begin{table}[!ht]",
    "  \\centering",
    sprintf("  \\caption{%s}", cap),
    sprintf("  \\label{%s}", lbl),
    "  \\resizebox{\\linewidth}{!}{%",
    sprintf("  \\begin{tabular}{@{}l%s@{}}", strrep("c", n_cols)),
    "  \\toprule",
    sprintf("  \\textbf{Method/Clf} & %s \\\\", hdr),
    "  \\midrule"
  )
  
  prev_meth <- ""
  for (i in seq_len(n_rows)) {
    lbl_parts <- strsplit(as.character(rw$MethodClf[i]), "/")[[1]]
    meth_part <- lbl_parts[1]
    clf_part  <- if (length(lbl_parts) > 1) lbl_parts[2] else ""
    
    if (meth_part != prev_meth && i > 1)
      lines <- c(lines, "  \\midrule")
    prev_meth <- meth_part
    
    ds_cells <- sapply(ds_cols, function(dc) {
      rv <- rw[[dc]][i]
      if (is.na(rv)) return("--")
      rv_fmt <- if (rv == floor(rv)) as.character(as.integer(rv))
      else sprintf("%.1f", rv)
      if (rv <= 1.5) sprintf("\\textbf{%s}", rv_fmt) else rv_fmt
    })
    
    mr_val  <- rw$MeanRank[i]
    mr_fmt  <- sprintf("%.2f", mr_val)
    if (!is.na(mr_val) && mr_val == min(rw$MeanRank, na.rm = TRUE))
      mr_fmt <- sprintf("\\textbf{%s}", mr_fmt)
    
    or_val  <- rw$OverallRank[i]
    or_fmt  <- if (!is.na(or_val) && or_val == floor(or_val))
      as.character(as.integer(or_val))
    else sprintf("%.1f", or_val)
    if (!is.na(or_val) && or_val <= 1.5)
      or_fmt <- sprintf("\\textbf{%s}", or_fmt)
    
    row_label <- if (meth_part == "MICE-GLMM-Stack")
      "\\textbf{MICE-GLMM-Stack}/All"
    else
      sprintf("%s/%s", gsub("_", "-", meth_part), clf_part)
    
    all_cells <- c(ds_cells, mr_fmt, or_fmt)
    lines <- c(lines,
               sprintf("  %s & %s \\\\", row_label,
                       paste(all_cells, collapse = " & ")))
  }
  
  kw_str <- sprintf(
    "  \\multicolumn{%d}{l}{\\footnotesize Kendall's $W = %.4f$; " %.%
      "$\\chi^2 = %.3f$; $df = %d$; $p = %s$} \\\\",
    n_cols + 1L,
    ifelse(is.na(kw$W), 0, kw$W),
    ifelse(is.na(kw$chi2), 0, kw$chi2),
    kw$df,
    ifelse(is.na(kw$p_value), "NA",
           ifelse(kw$p_value < 0.001, "<0.001",
                  sprintf("%.4f", kw$p_value)))
  )
  
  lines <- c(lines,
             "  \\bottomrule",
             kw_str,
             "  \\end{tabular}%",
             "  }",
             "\\end{table}",
             "")
  paste(lines, collapse = "\n")
}
cat("\n=== Kendall's W Rank Tables (MCAR + MAR, AUC + Brier) ===\n")
RANK_METRICS <- list(
  AUC   = TRUE,
  Brier = FALSE
)
rank_table_list  <- list()
kendall_summary_rows <- list()
for (mech in MISS_MECHS) {
  mech_data <- combined %>% dplyr::filter(MissMech == mech)
  
  for (met in names(RANK_METRICS)) {
    hib <- RANK_METRICS[[met]]
    cat(sprintf("\n------ %s / %s (rank 1 = %s) ------\n",
                mech, met, if (hib) "highest" else "lowest"))
    
    for (mr in MISS_RATES) {
      mr_pct <- as.integer(round(mr * 100))
      cat(sprintf("\n-- %d%% %s --\n", mr_pct, mech))
      
      rt <- build_rank_table(mech_data, mr,
                             metric          = met,
                             higher_is_better = hib,
                             mech            = mech)
      if (is.null(rt)) next
      rank_table_list[[paste0(mech, "_", met, "_", mr_pct)]] <- rt
      
      cat(sprintf("Rank table (%s, %s, %d%%) -- sorted by Overall Rank:\n",
                  mech, met, mr_pct))
      print(as.data.frame(rt$rank_wide[, c("MethodClf", rt$datasets,
                                           "MeanRank", "OverallRank")]),
            row.names = FALSE, digits = 2)
      
      kw <- rt$kendall
      cat(sprintf(
        "\nKendall's W = %.4f  |  chi2(%d) = %.3f  |  p-value = %s\n",
        ifelse(is.na(kw$W), NaN, kw$W),
        kw$df,
        ifelse(is.na(kw$chi2), NaN, kw$chi2),
        ifelse(is.na(kw$p_value), "NA",
               ifelse(kw$p_value < 0.001, "< 0.001",
                      sprintf("%.4f", kw$p_value)))
      ))
      
      kendall_summary_rows[[length(kendall_summary_rows) + 1]] <- data.frame(
        Metric       = met,
        MissMech     = mech,
        MissRate     = mr,
        MissRate_pct = mr_pct,
        W            = ifelse(is.na(kw$W),    NA_real_, kw$W),
        chi2         = ifelse(is.na(kw$chi2), NA_real_, kw$chi2),
        df           = kw$df,
        p_value      = ifelse(is.na(kw$p_value), NA_real_, kw$p_value),
        n_methods    = nrow(rt$rank_mat),
        n_datasets   = ncol(rt$rank_mat),
        stringsAsFactors = FALSE
      )
      
      csv_rt <- rt$rank_wide
      csv_rt$MissRate_pct <- mr_pct
      csv_rt$Metric       <- met
      csv_rt$MissMech     <- mech
      csv_file <- file.path(OUTPUT_DIR,
                            sprintf("Multi_HAS_rank_table_%s_%02dpct_%s.csv", mech, mr_pct, met))
      write.csv(csv_rt, csv_file, row.names = FALSE)
      cat(sprintf("Saved: %s\n", csv_file))
    }
  }
}
if (length(kendall_summary_rows) > 0) {
  kendall_df <- do.call(rbind, kendall_summary_rows)
  rownames(kendall_df) <- NULL
  kw_csv <- file.path(OUTPUT_DIR, "Multi_HAS_kendall_W_summary.csv")
  write.csv(kendall_df, kw_csv, row.names = FALSE)
  cat(sprintf("\nKendall's W summary saved: %s\n", kw_csv))
  cat("\n=== Kendall's W -- all mechanisms, all levels, both metrics ===\n")
  print(kendall_df[, c("MissMech","Metric","MissRate_pct","W","chi2","df","p_value")],
        row.names = FALSE, digits = 4)
}
for (mech in MISS_MECHS) {
  for (met in names(RANK_METRICS)) {
    keys      <- paste0(mech, "_", met, "_", as.integer(MISS_RATES * 100))
    tex_lines <- latex_preamble(9)
    tex_lines <- c(tex_lines,
                   sprintf("%% ============================================================"),
                   sprintf("%% Kendall's W Rank Tables -- %s, %s, one per missingness level", mech, met),
                   sprintf("%% ============================================================"),
                   "")
    
    for (k in keys) {
      if (!is.null(rank_table_list[[k]]))
        tex_lines <- c(tex_lines, rank_table_to_latex(rank_table_list[[k]]))
    }
    
    tex_lines <- c(tex_lines, "\\end{document}")
    tex_file  <- file.path(OUTPUT_DIR,
                           sprintf("Multi_HAS_rank_tables_%s_%s.tex", mech, met))
    writeLines(tex_lines, tex_file)
    cat(sprintf("Saved: %s\n", tex_file))
  }
}
clf_ltype6 <- c(J48="solid", NB="dashed", PART="dotted",
                SMO="twodash", RF="dotdash", All="solid")
clf_shape6 <- c(J48=16L, NB=17L, PART=15L, SMO=18L, RF=19L, All=8L)
for (mech in MISS_MECHS) {
  rank_trend_rows <- list()
  for (k in names(rank_table_list)) {
    rt <- rank_table_list[[k]]
    if (is.null(rt) || rt$metric != "AUC" || rt$mech != mech) next
    rw <- rt$rank_wide
    rw$MissRate  <- rt$miss_rate
    rw$MethodClf <- as.character(rw$MethodClf)
    rw$Method    <- sub("/.*", "", rw$MethodClf)
    rank_trend_rows[[length(rank_trend_rows)+1]] <-
      rw[, c("MethodClf","Method","MissRate","MeanRank")]
  }
  
  if (length(rank_trend_rows) == 0) next
  
  mech_desc <- unname(mech_labels[mech]); if (is.na(mech_desc) || is.null(mech_desc)) mech_desc <- mech
  
  rank_trend <- do.call(rbind, rank_trend_rows) %>%
    dplyr::mutate(
      Method      = factor(Method, levels = METHOD_ORDER),
      MethodLabel = factor(dplyr::recode(as.character(Method), !!!METHOD_LABELS),
                           levels = unname(METHOD_LABELS)),
      Clf         = sub(".*/", "", MethodClf),
      Clf         = factor(Clf, levels = c(CLF_NAMES, "All"))) %>%
    dplyr::filter(!is.na(MethodLabel), Method != "Complete")
  
  rank_trend_comp  <- rank_trend %>% dplyr::filter(Method != "MICE-GLMM-Stack")
  rank_trend_stack <- rank_trend %>% dplyr::filter(Method == "MICE-GLMM-Stack")
  
  p8 <- ggplot(rank_trend_comp,
               aes(x = MissRate * 100, y = MeanRank,
                   colour   = MethodLabel,
                   linetype = Clf,
                   shape    = Clf,
                   group    = MethodClf)) +
    geom_line(linewidth = 0.65) +
    geom_point(size = 1.7) +
    geom_line(data = rank_trend_stack,
              aes(x = MissRate * 100, y = MeanRank, group = MethodClf),
              colour = "#43A047", linewidth = 1.5, linetype = "solid",
              inherit.aes = FALSE) +
    geom_point(data = rank_trend_stack,
               aes(x = MissRate * 100, y = MeanRank),
               colour = "#43A047", shape = 8L, size = 2.5,
               inherit.aes = FALSE) +
    facet_wrap(~ MethodLabel, nrow = 2) +
    scale_y_reverse(breaks = seq(2, ceiling(max(rank_trend$MeanRank, na.rm = TRUE)), 4)) +
    scale_colour_manual(values = setNames(colour_pal, METHOD_LABELS),
                        name = "Method") +
    scale_linetype_manual(values = clf_ltype6, name = "Classifier") +
    scale_shape_manual(   values = clf_shape6, name = "Classifier") +
    scale_x_continuous(breaks = MISS_RATES * 100,
                       labels = paste0(MISS_RATES * 100, "%")) +
    labs(title   = sprintf("Mean AUC Rank Across Missingness Rates: All Method x Classifier Competitors Under %s", mech_desc),
         caption = "Each line: one Method/Classifier combination. Bold green line: MICE-GLMM-Stack (proposed). Rank 1 = best. Averaged over fifteen benchmark datasets.",
         x = "Missingness Rate (%)", y = "Mean Rank (1 = Best)") +
    theme_paper() +
    guides(
      colour   = guide_legend(nrow = 2,
                              override.aes = list(linewidth = 1.2)),
      linetype = guide_legend(nrow = 1),
      shape    = guide_legend(nrow = 1))
  
  fig8_file <- sprintf("fig8_rank_trend_%s", mech)
  ggsave(file.path(OUTPUT_DIR, paste0(fig8_file, ".png")),
         plot = p8, width = 16, height = 7, units = "in", dpi = 900)
  ggsave(file.path(OUTPUT_DIR, paste0(fig8_file, ".pdf")),
         plot = p8, width = 16, height = 7, units = "in", device = cairo_pdf)
  cat(sprintf("Saved: %s (.png 900 dpi + .pdf)\n", fig8_file))
}
if (exists("varcomp_combined") && !is.null(varcomp_combined) && nrow(varcomp_combined) > 0) {
  vc_plot_dat <- varcomp_combined %>%
    dplyr::filter(Component != "Classifier -- slope/intercept covariance") %>%
    dplyr::mutate(RungLabel = factor(
      paste0("Rung ", FormulaRung),
      levels = paste0("Rung ", sort(unique(FormulaRung)))))
  
  p9 <- ggplot(vc_plot_dat, aes(x = Component, y = sdcor, fill = Component)) +
    geom_boxplot(outlier.size = 0.6, linewidth = 0.35, width = 0.65) +
    facet_grid(MissMech ~ RungLabel, scales = "free_x") +
    scale_fill_brewer(palette = "Set2") +
    labs(title   = "Empirical Variance-Component Estimates: Pooled MICE-GLMM-Stack Random Effects",
         caption = "Each box: link-scale SD (or slope/intercept correlation) of one random-effects component, over every converged train-test split. Faceted by missingness mechanism (rows) and which fallback rung (Sec.2.4) converged (columns).",
         x = NULL, y = "Estimated SD (link scale)") +
    theme_paper() +
    theme(axis.text.x   = element_text(angle = 35, hjust = 1, size = 6.5),
          legend.position = "none")
  
  ggsave(file.path(OUTPUT_DIR, "fig9_variance_components.png"),
         plot = p9, width = 14, height = 6 * length(unique(vc_plot_dat$MissMech)),
         units = "in", dpi = 900, limitsize = FALSE)
  ggsave(file.path(OUTPUT_DIR, "fig9_variance_components.pdf"),
         plot = p9, width = 14, height = 6 * length(unique(vc_plot_dat$MissMech)),
         units = "in", device = cairo_pdf, limitsize = FALSE)
  cat("Saved: fig9_variance_components (.png 900 dpi + .pdf)\n")
} else {
  cat("[WARN] No variance-component data captured -- skipping fig9_variance_components.\n")
}
cat("\n=== ALL DONE (incl. Kendall's W rank tables) ===\n")
