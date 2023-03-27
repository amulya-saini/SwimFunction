suppressPackageStartupMessages(library("sigclust2"))
library("tidyr")
library("ggplot2")
library("WGCNA")

# I forked and modified the original sigclust2
# to use Spearman correlation instead of Pearson
# If I change the forked copy, run the two lines below.
# > remove.packages("sigclust2")
# > devtools::install_github("discolemur/sigclust2")

DEBUG <- FALSE

no_change <- function(x) {
  return(as.dist(x))
}

get_shc_and_distances <- function(data, metric, linkage_method) {
  # Parameters:
  #     data (data frame)
  #     metric (precomputed, cor [interpreted as spearman], euclidean, etc.)
  #     linkage (complete, average, etc.)
  x <- as.matrix(data)
  l <- 2
  matmet <- NULL
  if (metric == "precomputed") {
    matmet <- no_change
  }

  shc_result <- sigclust2::shc(
    x = x,
    metric = metric,
    matmet = matmet,
    linkage = linkage_method,
    l = l,
    rcpp = FALSE)

  distances <- NULL
  if (metric == "precomputed") {
    distances <- x
  } else if (metric == "cor") {
    distances <- 1 - WGCNA::cor(t(x), method = "spearman")
  } else {
    distances <- dist(x, method = metric, p = l)
  }

  return(
    list(
      shc_result = shc_result,
      distances = distances))
}

get_num_observed <- function(merge) {
  observed <- rep(NA, length(merge[, 1]))
  for (i in seq_along(merge[, 1])) {
    a <- 0
    b <- 0
    if (merge[i, 1] < 0) {
      a <- 1
    }  else {
      a <- observed[merge[i, 1]]
    }
    if (merge[i, 2] < 0) {
      b <- 1
    } else {
      b <- observed[merge[i, 2]]
    }
    observed[i] <- a + b
  }
  return(observed)
}

get_linkage_dataframe <- function(shc_result) {
  # Convert to scipy.cluster.hierarchy.linkage compatible Z matrix
  indices <- (shc_result$hc_dat$merge * -1) - 1
  # add 29
  indices[indices < 0] <- (indices[indices < 0] * -1) + (
    length(shc_result$hc_dat$merge[, 1]) - 1)
  linkage <- data.frame(
    i1 = indices[, 1],
    i2 = indices[, 2],
    heights = shc_result$hc_dat$height,
    obs = get_num_observed(shc_result$hc_dat$merge)
  )
  return(linkage)
}

save_linkage_and_pvals <- function(
    shc_dist_list, distance_csv_file, linkage_csv_file, ordered_labels_csv_file,
    plot_fname, width, height) {

  # Convert to scipy.cluster.hierarchy.linkage compatible Z matrix
  linkage <- get_linkage_dataframe(shc_dist_list$shc_result)
  linkage["pvals"] <- rev(shc_dist_list$shc_result$p_norm)
  plot_data <- plot(shc_dist_list$shc_result, hang = .1)

  # Write ordered labels
  ordered_labels <- shc_dist_list$shc_result$hc_dat$labels[
        shc_dist_list$shc_result$hc_dat$order]
  writeLines(unlist(lapply(
    ordered_labels, paste, collapse = ",")), ordered_labels_csv_file)

  # Write distances
  write.csv(
    as.data.frame(as.matrix(shc_dist_list$distances)),
    distance_csv_file,
    row.names = FALSE,
    col.names = FALSE
  )
  # Write linkage and significance
  write.csv(linkage, linkage_csv_file)
  # Write plot
  ggplot2::ggsave(
    plot_fname,
    plot_data,
    width = width,
    height = height,
    dpi = 200
  )
}

cluster_rows <- function(data, metric, linkage_method, outdir) {
  # Cluster rows
  row_linkage_csv <- file.path(
    outdir, "CLUSTERING_all_metrics_row_linkage.csv")
  row_dist_csv <- file.path(
    outdir, "CLUSTERING_all_metrics_row_dist.csv")
  row_metric_png <- file.path(
    outdir, "CLUSTERING_all_metrics_row_dendrogram.png")
  ordered_labels_csv <- file.path(
    outdir, "CLUSTERING_all_metrics_row_ordered_labels.csv")
  save_linkage_and_pvals(
    get_shc_and_distances(data, metric, linkage_method),
    row_dist_csv,
    row_linkage_csv,
    ordered_labels_csv,
    row_metric_png,
    width = 45,
    height = 7
  )
}

cluster_columns <- function(data, metric, linkage_method, outdir) {
  # Cluster columns
  column_dist_csv <- file.path(
    outdir, "CLUSTERING_all_metrics_col_dist.csv")
  column_linkage_csv <- file.path(
    outdir, "CLUSTERING_all_metrics_col_linkage.csv")
  column_metric_png <- file.path(
    outdir, "CLUSTERING_all_metrics_col_dendrogram.png")
  ordered_labels_csv <- file.path(
    outdir, "CLUSTERING_all_metrics_col_ordered_labels.csv")
  save_linkage_and_pvals(
    get_shc_and_distances(t(data), metric, linkage_method),
    column_dist_csv,
    column_linkage_csv,
    ordered_labels_csv,
    column_metric_png,
    width = 8,
    height = 8
  )
}

cluster_data <- function(metric, linkage_method, filename) {
  outdir <- dirname(normalizePath(filename))

  data <- read.csv(file = filename, header = TRUE)
  row.names(data) <- data[, 1]
  data <- data[, -1]
  data <- tidyr::drop_na(data)
  print(colnames(data))

  cluster_rows(data, metric, linkage_method, outdir)
  cluster_columns(data, metric, linkage_method, outdir)
}

parse_args <- function() {
  if (!DEBUG) {
    args <- commandArgs(trailingOnly = TRUE)
  } else {
    args <- c("/tmp/CLUSTERING_all_metrics_scaled.csv")
  }
  if (length(args) < 1) {
    stop("Usage: Rscript {file}.R path_to_data.csv [-l linkage] [-m metric]",
      call. = FALSE)
  }
  filename <- args[1]
  linkage_method <- "complete"
  metric <- "cor"
  metric_loc <- match("-m", args)
  if (!is.na(metric_loc)) {
    metric <- args[metric_loc + 1]
  }
  linkage_loc <- match("-l", args)
  if (!is.na(linkage_loc)) {
    linkage_method <- args[linkage_loc + 1]
  }
  cat(c(
    "Using metric \"",
    metric,
    "\" and linkage \"",
    linkage_method,
    "\"\n"), sep = "")
  return(list(
    filename = filename,
    metric = metric,
    linkage_method = linkage_method
  ))
}

args_ <- parse_args()
cluster_data(args_$metric, args_$linkage_method, args_$filename)
