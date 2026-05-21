library(hoopR)
library(progressr)
library(tictoc)

tictoc::tic()

progressr::with_progress({
  nba_pbp <- hoopR::load_nba_pbp()
})

tictoc::toc()

# Check what we got
cat("\nData loaded successfully!\n")
cat(sprintf("Dimensions: %d rows x %d columns\n", nrow(nba_pbp), ncol(nba_pbp)))
cat(sprintf("Column names: %s\n", paste(head(names(nba_pbp), 10), collapse = ", ")))
if (length(names(nba_pbp)) > 10) cat("...\n")
cat("\nFirst few rows:\n")
print(head(nba_pbp, 3))

# Save to CSV file (optional)
output_file <- "nba_pbp_hoopR.csv"
cat(sprintf("\nSaving data to %s...\n", output_file))
write.csv(nba_pbp, file = output_file, row.names = FALSE)
cat(sprintf("✓ Data saved to %s\n", output_file))

