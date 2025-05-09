# Load required packages
# install.packages("hoopR") # Uncomment if you haven't installed it
# install.packages("progressr") # Uncomment if you haven't installed it
# install.packages("tictoc") # Uncomment if you haven't installed it
# install.packages("futile.logger") # Uncomment if you haven't installed it
library(hoopR)
library(progressr)
library(tictoc)
library(futile.logger)
library(dplyr)

# Configure logging
flog.appender(appender.file("nba_data_fetch.log"))
flog.info("Starting NBA data fetch script")

# Function to create a retry wrapper
retry_api_call <- function(func, max_attempts = 3, delay_seconds = 5, ...) {
  attempt <- 1
  while (attempt <= max_attempts) {
    result <- tryCatch({
      func(...)
    }, error = function(e) {
      flog.warn(sprintf("Attempt %d/%d failed: %s", attempt, max_attempts, e$message))
      if (attempt == max_attempts) {
        flog.error(sprintf("All %d attempts failed for API call", max_attempts))
        return(NULL)
      }
      Sys.sleep(delay_seconds)
      return("RETRY")
    })
    
    if (!identical(result, "RETRY")) {
      return(result)
    }
    
    attempt <- attempt + 1
  }
  return(NULL)
}

# Create fancy banner
create_banner <- function(text) {
  banner_width <- nchar(text) + 8
  border <- paste(rep("=", banner_width), collapse = "")
  cat("\n", border, "\n")
  cat("==  ", text, "  ==\n")
  cat(border, "\n\n")
}

# Define the target season (use the starting year) and output file
# For the 2024-2025 season (which includes May 2, 2025), the starting year is 2024.
target_year <- 2024
output_file <- paste0("NBA",target_year + 1, ".csv")

create_banner(sprintf("NBA Play-by-Play Data Fetcher - Season %d-%d", target_year, target_year + 1))

# Initialize progress handlers
handlers(global = TRUE)
handlers(handler_progress(
  format = "  [:bar] :percent | ETA: :eta | Elapsed: :elapsed",
  width = 60,
  complete = "=",
  incomplete = "-"
))

# 1) Get game log for the target season with retry
gl_data <- NULL
cat("Step 1/3: Fetching season game log...\n")
gl_data <- retry_api_call(
  func = hoopR::nba_leaguegamelog,
  max_attempts = 3,
  league_id = "00", 
  season = as.character(target_year)
)

# Check if game log fetching was successful and returned data
if (!is.null(gl_data) && length(gl_data) > 0 && inherits(gl_data[[1]], "data.frame") && nrow(gl_data[[1]]) > 0) {
  df <- gl_data[[1]] # Extract the dataframe
  cat(sprintf("✓ Successfully fetched game log with %d entries.\n", nrow(df)))
  flog.info(sprintf("Game log fetched with %d entries", nrow(df)))

  # 2) Extract unique game IDs
  cat("Step 2/3: Extracting game IDs...\n")
  game_ids <- NULL
  if ("GAME_ID" %in% names(df)) {
     game_ids <- unique(df$GAME_ID)
  } else if (ncol(df) >= 5) {
     # Fallback to column index 5 if GAME_ID column name doesn't exist
     flog.warn("'GAME_ID' column not found in game log, attempting to use column 5.")
     game_ids <- unique(df[[5]])
  } else {
     flog.error("Could not find 'GAME_ID' column or column 5 in the game log data.")
  }

  # Check if game IDs were extracted successfully
  if (!is.null(game_ids) && length(game_ids) > 0) {
    cat(sprintf("✓ Found %d unique game IDs for season %d-%d\n", 
                length(game_ids), target_year, target_year + 1))
    flog.info(sprintf("Found %d unique game IDs", length(game_ids)))

    # 3) Fetch play-by-play with progress bar for individual games
    cat("Step 3/3: Fetching play-by-play data for each game...\n")
    tictoc::tic("Total PBP fetch time")
    
    # Create progress bar for individual games
    p <- progressr::progressor(steps = length(game_ids))
    
    # Initialize container for successful PBP data
    pbp_all_games <- list()
    successful_games <- 0
    failed_games <- 0
    
    # Process games one by one with retry logic
    for (i in seq_along(game_ids)) {
      game_id <- game_ids[i]
      # Update progress bar with game number
      p(sprintf("Game %d/%d (ID: %s)", i, length(game_ids), game_id))
      
      # Fetch PBP with retry
      pbp_single_game <- retry_api_call(
        func = hoopR::nba_pbp,
        max_attempts = 3,
        delay_seconds = 2,
        game_id = game_id,
        on_court = TRUE
      )
      
      # Check if we got data for this game
      if (!is.null(pbp_single_game) && inherits(pbp_single_game, "data.frame") && nrow(pbp_single_game) > 0) {
        pbp_all_games[[i]] <- pbp_single_game
        successful_games <- successful_games + 1
        flog.info(sprintf("Successfully fetched game %s (%d/%d)", game_id, i, length(game_ids)))
      } else {
        failed_games <- failed_games + 1
        flog.warn(sprintf("Failed to fetch game %s after retries (%d/%d)", game_id, i, length(game_ids)))
      }
      
      # Small pause to avoid rate limiting
      Sys.sleep(0.2)
    }
    
    # Combine all successful game data
    if (length(pbp_all_games) > 0) {
      pbp_data <- bind_rows(pbp_all_games)
      tictoc::toc()
      
      # Display summary
      cat(sprintf("\n✓ Processed %d games: %d successful, %d failed\n", 
                 length(game_ids), successful_games, failed_games))
      
      if (failed_games > 0) {
        cat(sprintf("⚠ %d games could not be fetched. Check log file for details.\n", failed_games))
      }
      
      # 4) Write CSV if PBP data was fetched successfully
      if (nrow(pbp_data) > 0) {
        cat(sprintf("Writing data to %s...\n", output_file))
        tryCatch({
          write.csv(pbp_data, file = output_file, row.names = FALSE)
          cat(sprintf("✓ Successfully wrote PBP data with %d rows to %s\n", nrow(pbp_data), output_file))
          flog.info(sprintf("Wrote %d rows to %s", nrow(pbp_data), output_file))
        }, error = function(e) {
          cat(sprintf("❌ Error writing CSV file '%s': %s\n", output_file, e$message))
          flog.error(sprintf("Error writing CSV: %s", e$message))
        })
      } else {
        cat("❌ No PBP data was collected. Nothing to write.\n")
        flog.error("No PBP data was collected")
      }
    } else {
      cat("❌ Failed to fetch PBP data for any games.\n")
      flog.error("Failed to fetch PBP data for any games")
    }
  } else {
    cat("❌ No game IDs were successfully extracted from the game log.\n")
    flog.error("No game IDs were extracted from the game log")
  }
} else {
  cat("❌ Failed to fetch game log for the season.\n")
  flog.error("Failed to fetch game log for the season")
}

create_banner("NBA Data Fetching Process Complete")

# Print summary statistics
if (exists("pbp_data") && !is.null(pbp_data)) {
  cat(sprintf("Total rows fetched: %d\n", nrow(pbp_data)))
  cat(sprintf("Game success rate: %.1f%% (%d/%d)\n", 
              successful_games/length(game_ids)*100, successful_games, length(game_ids)))
  cat(sprintf("Data saved to: %s\n", output_file))
  cat(sprintf("Log file: %s\n", "nba_data_fetch.log"))
}

flog.info("Script execution completed")