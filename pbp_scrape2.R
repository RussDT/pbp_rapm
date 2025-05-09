# Load required packages
# install.packages("hoopR") # Uncomment if you haven't installed it
# install.packages("progressr") # Uncomment if you haven't installed it
# install.packages("tictoc") # Uncomment if you haven't installed it
library(hoopR)
library(progressr)
library(tictoc)

# Define the target season (use the starting year) and output file
# For the 2024-2025 season (which includes May 2, 2025), the starting year is 2024.
target_year <- 2024
output_file <- "NBA25.csv"

message("=== Fetching data for Season ", target_year, "-", target_year + 1, " (output: ", output_file, ") ===")

# 1) Get game log for the target season
# Using tryCatch to handle potential errors during API calls
gl_data <- NULL
progressr::with_progress({
  gl_data <- tryCatch({
    hoopR::nba_leaguegamelog(league_id = "00", season = as.character(target_year))
  }, error = function(e) {
    message("Error fetching game log for season ", target_year, ": ", e$message)
    return(NULL) # Return NULL on error
  })
})

# Check if game log fetching was successful and returned data
# hoopR::nba_leaguegamelog usually returns a list, check the first element
if (!is.null(gl_data) && length(gl_data) > 0 && inherits(gl_data[[1]], "data.frame") && nrow(gl_data[[1]]) > 0) {

  df <- gl_data[[1]] # Extract the dataframe
  message("Successfully fetched game log with ", nrow(df), " entries.")

  # 2) Extract unique game IDs
  # Preferably use column name 'GAME_ID'. Fallback to index 5 if needed.
  game_ids <- NULL
  if ("GAME_ID" %in% names(df)) {
     game_ids <- unique(df$GAME_ID)
  } else if (ncol(df) >= 5) {
     # Fallback to column index 5 if GAME_ID column name doesn't exist
     message("Warning: 'GAME_ID' column not found in game log, attempting to use column 5.")
     game_ids <- unique(df[[5]])
  } else {
     message("Error: Could not find 'GAME_ID' column or column 5 in the game log data.")
  }

  # Check if game IDs were extracted successfully
  if (!is.null(game_ids) && length(game_ids) > 0) {
    message("Found ", length(game_ids), " unique game IDs for season ", target_year)

    # 3) Fetch play-by-play and time it
    pbp_data <- NULL
    tictoc::tic(paste("   Fetching PBP data for", length(game_ids), "games in season", target_year))
    progressr::with_progress({
       pbp_data <- tryCatch({
          # Fetch PBP data for the extracted game IDs
          hoopR::nba_pbps(game_ids = game_ids, on_court = TRUE)
       }, error = function(e) {
          message("Error fetching PBP data: ", e$message)
          return(NULL) # Return NULL on error
       })
    })
    tictoc::toc()

    # 4) Write CSV if PBP data was fetched successfully
    if (!is.null(pbp_data)) {
      tryCatch({
        write.csv(pbp_data, file = output_file, row.names = FALSE)
        message("Successfully wrote PBP data to ", output_file)
      }, error = function(e) {
        message("Error writing CSV file '", output_file, "': ", e$message)
      })
    } else {
       message("Skipping CSV write because PBP data fetching failed or returned no data.")
    }
  } else {
     message("Skipping PBP fetch because no game IDs were successfully extracted from the game log.")
  }
} else {
  message("Skipping PBP fetch because game log fetching failed or returned no data/unexpected format.")
}

message("=== Data fetching process finished. ===")