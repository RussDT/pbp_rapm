# load required packages
library(hoopR)
library(progressr)
library(tictoc)

# vector of seasons and a helper to format two-digit indices
seasons <- 1999:2000
idx <- seq_along(seasons) - 1L  # 0 for 1999, 1 for 2000, …, 25 for 2024

for (i in seq_along(seasons)) {
  yr   <- seasons[i]
  tag  <- sprintf("%02d", idx[i])        # "00", "01", …, "25"
  file <- sprintf("NBA%s.csv", tag)      # "NBA00.csv", …

  message("=== Season ", yr, " (output: ", file, ") ===")

  # 1) get game log
  progressr::with_progress({
    gl <- hoopR::nba_leaguegamelog(league_id = "00", season = as.character(yr))
  })
  df <- as.data.frame(gl)

  # 2) extract unique game IDs (column 5 in your original code)
  game_ids <- unique(df[[5]])

  # 3) fetch play-by-play and time it
  tictoc::tic(paste("  pbp", yr))
  progressr::with_progress({
    pbp <- hoopR::nba_pbps(game_ids = game_ids, on_court = TRUE)
  })
  tictoc::toc()

  # 4) write CSV and sleep
  write.csv(pbp, file = file, row.names = FALSE)
  Sys.sleep(3)
}