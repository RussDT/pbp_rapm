library(dplyr)
library(stringr)
library(tidyr)
library(zoo)

##Rebounding
ProcessRebounding <- function(file_name, year) {
NBA24 = read.csv(file_name) 
NBA24 <- NBA24 %>%
  mutate(event_type = case_when(
    event_type == 1 ~ "MAKE",
    event_type == 2 ~ "MISS",
    event_type == 3 ~ "FreeThrow",
    event_type == 4 ~ "Rebound",
    event_type == 5 ~ "Turnover",
    event_type == 6 ~ "Foul",
    event_type == 7 ~ "Violation",
    event_type == 8 ~ "Substitution",
    event_type == 9 ~ "Timeout",
    event_type == 10 ~ "JumpBall",
    event_type == 11 ~ "Ejection",
    event_type == 12 ~ "StartOfPeriod",
    event_type == 13 ~ "EndOfPeriod",
    event_type == 14 ~ "Empty",
    TRUE ~ as.character(event_type)  # Default case if none of the above
  ))



# Replace NA values with empty strings in the entire dataframe
NBA24 <- NBA24 %>%
  mutate(across(where(is.character), ~replace_na(., "")))


NBA24 <- NBA24  %>%
  mutate(
    Home_Action_Score = case_when(
      grepl("Free Throw", home_description, ignore.case = TRUE) & !grepl("MISS", home_description, ignore.case = TRUE) ~ 1,
      grepl("PTS", home_description, ignore.case = TRUE) & !grepl("3PT", home_description, ignore.case = TRUE) & event_type == "MAKE" ~ 2,
      grepl("3PT", home_description, ignore.case = TRUE) & !grepl("MISS", home_description, ignore.case = TRUE) ~ 3,
      TRUE ~ 0
    ),
    Away_Action_Score = case_when(
      grepl("Free Throw", visitor_description, ignore.case = TRUE) & !grepl("MISS", visitor_description, ignore.case = TRUE) ~ 1,
      grepl("PTS", visitor_description, ignore.case = TRUE) & !grepl("3PT", visitor_description, ignore.case = TRUE) & event_type == "MAKE" ~ 2,
      grepl("3PT", visitor_description, ignore.case = TRUE) & !grepl("MISS", visitor_description, ignore.case = TRUE) ~ 3,
      TRUE ~ 0
    ),
    Net_Home = cumsum(Home_Action_Score),
    Net_Away = cumsum(Away_Action_Score),
    Net_Total = Net_Home + Net_Away
  )
rstudioapi::writeRStudioPreference("data_viewer_max_columns", 1000L)



NBA24 <- NBA24 %>%
  mutate(
    prev_seconds = lag(seconds_remaining_quarter, 1),
    Prev_visitor_desc = lag(visitor_description, 1, default = ""),  # visitor description from 1 row back
    Prev_visitor_desc2 = lag(visitor_description, 2, default = ""),
    Prev_home_desc = lag(home_description, 1, default = ""),  # home description from 1 row back
    Prev_home_desc2 = lag(home_description, 2, default = ""),
    Next_home_desc = lead(home_description, default = ""),  # Shift home descriptions down
    Next_event = lead(event_type, default = ""),  # Shift home descriptions down
    Next_visitor_desc = lead(visitor_description, default = ""),  # Shift visitor descriptions down
    Offensive_Rebound = case_when(
      str_detect(home_description, "REBOUND") & 
        str_detect(Prev_home_desc, "MISS") & (!str_detect(time_quarter, "0:00") & (!str_detect(Next_event,"EndOfPeriod" ))) & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ 1,
      
      str_detect(home_description, "Rebound") & 
        str_detect(Prev_home_desc, "MISS") & (!str_detect(time_quarter, "0:00") & (!str_detect(Next_event,"EndOfPeriod" ))) & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))  ~ 1,
      
      str_detect(home_description, "REBOUND") & 
        str_detect(Prev_home_desc, "Putback") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))  ~ 1,
      
      str_detect(visitor_description, "REBOUND") & 
        str_detect(Prev_visitor_desc, "MISS") & (!str_detect(time_quarter, "0:00") & (!str_detect(Next_event,"EndOfPeriod" ))) & 
            (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))  ~ 1,
      
      str_detect(visitor_description, "Rebound") & 
        str_detect(Prev_visitor_desc, "MISS") & (!str_detect(time_quarter, "0:00") & (!str_detect(Next_event,"EndOfPeriod" ))) & 
            (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))  ~ 1,
      
      str_detect(visitor_description, "Rebound") & 
        str_detect(Prev_visitor_desc, "Putback") & (!str_detect(time_quarter, "0:00") & (!str_detect(Next_event,"EndOfPeriod" ))) & 
            (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ 1,
      TRUE ~ FALSE
    ),
    End_of_Possession = case_when(
      str_detect(home_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
      str_detect(home_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
      str_detect(visitor_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
      str_detect(visitor_description, "Rebound") & str_detect(Prev_home_desc, "MISS") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
      str_detect(home_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
      str_detect(home_description, "Rebound") & str_detect(Prev_home_desc, "MISS") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
      str_detect(visitor_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
      str_detect(visitor_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS") & 
            (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
      TRUE ~ FALSE  # Default case to FALSE
    ),     
    TeamOnOffense = case_when(
      (str_detect(home_description, "Flagrant 2 of 2") | str_detect(home_description, "Flagrant 3 of 3"))  ~ "Home",
      (str_detect(visitor_description, "Flagrant 2 of 2") | str_detect(visitor_description, "Flagrant 3 of 3"))  ~ "Away",
      str_detect(home_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS")  ~ "Away",
      str_detect(home_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS")  ~ "Away",
      str_detect(visitor_description, "REBOUND") & str_detect(Prev_home_desc, "MISS")  ~ "Home",
      str_detect(visitor_description, "Rebound") & str_detect(Prev_home_desc, "MISS")  ~ "Home",
      str_detect(home_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") ~ "Home",
      str_detect(home_description, "Rebound") & str_detect(Prev_home_desc, "MISS") ~ "Home",
      str_detect(visitor_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") ~ "Away",
      str_detect(visitor_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS") ~ "Away",
      str_detect(home_description, "Turnover")  ~ "Home",
      str_detect(visitor_description, "Turnover")   ~ "Away",
      str_detect(home_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_visitor_desc, "S.FOUL")  ~ "Home",
      str_detect(visitor_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_home_desc, "S.FOUL")  ~ "Away",
      str_detect(home_description, "1 of 1") | str_detect(home_description, "2 of 2") | str_detect(home_description, "3 of 3")  ~ "Home",
      str_detect(visitor_description, "1 of 1") | str_detect(visitor_description, "2 of 2") | str_detect(visitor_description, "3 of 3") ~ "Away",
      TRUE ~ ""  # Default case to NA
    )
  ) 

NBA24 <- NBA24 %>%
  mutate(
    PotentialFoul = case_when(
      str_detect(event_type, "Foul") & !str_detect(visitor_description, "T.FOUL") & !str_detect(visitor_description, "FLAGRANT") & !str_detect(visitor_description, "Offens") &
        !str_detect(visitor_description, "Transition")  & !str_detect(home_description, "T.FOUL") & !str_detect(home_description, "Transition") & !str_detect(home_description, "FLAGRANT") ~ TRUE,
      TRUE ~ FALSE
    )
  )



# Show updated data frame


# Assuming your data frame is named df
NBA24 <- NBA24 %>%
  mutate(
    a1 = away_player1,
    a2 = away_player2,
    a3 = away_player3,
    a4 = away_player4,
    a5 = away_player5,
    h1 = home_player1,
    h2 = home_player2,
    h3 = home_player3,
    h4 = home_player4,
    h5 = home_player5
  )


NBA24 <- NBA24 %>%
  mutate(
    O1 = case_when(
      TeamOnOffense == "Away" ~ a1,
      TeamOnOffense == "Home" ~ h1,
      TRUE ~ 0
    ),
    O2 = case_when(
      TeamOnOffense == "Away" ~ a2,
      TeamOnOffense == "Home" ~ h2,
      TRUE ~ 0
    ),
    O3 = case_when(
      TeamOnOffense == "Away" ~ a3,
      TeamOnOffense == "Home" ~ h3,
      TRUE ~ 0
    ),
    O4 = case_when(
      TeamOnOffense == "Away" ~ a4,
      TeamOnOffense == "Home" ~ h4,
      TRUE ~ 0
    ),
    O5 = case_when(
      TeamOnOffense == "Away" ~ a5,
      TeamOnOffense == "Home" ~ h5,
      TRUE ~ 0
    ),
    D1 = case_when(
      TeamOnOffense == "Away" ~ h1,
      TeamOnOffense == "Home" ~ a1,
      TRUE ~ 0
    ),
    D2 = case_when(
      TeamOnOffense == "Away" ~ h2,
      TeamOnOffense == "Home" ~ a2,
      TRUE ~ 0
    ),
    D3 = case_when(
      TeamOnOffense == "Away" ~ h3,
      TeamOnOffense == "Home" ~ a3,
      TRUE ~ 0
    ),
    D4 = case_when(
      TeamOnOffense == "Away" ~ h4,
      TeamOnOffense == "Home" ~ a4,
      TRUE ~ 0
    ),
    D5 = case_when(
      TeamOnOffense == "Away" ~ h5,
      TeamOnOffense == "Home" ~ a5,
      TRUE ~ 0
    )
  )


progressr::with_progress({
  Player24 <- hoopR::nba_commonallplayers(league_id = '00', season = 2024)
})

Players = as.data.frame(Player24)


Dict24 <- Players[, c(1, 3)]


NBA24Filt <- NBA24[NBA24$End_of_Possession == TRUE, ]

print("Last Part")

for (i in 1:nrow(Dict24)) {
  value_to_replace <- Dict24[i,1]
  replacement_value <- Dict24[i,2]
  
  NBA24Filt[, 79:88] <- lapply(NBA24Filt[, 79:88], function(x) {
    replace(x, x == value_to_replace, replacement_value)
  })
}


View(NBA24Filt)

NBA24Filt  <- NBA24Filt  %>%
  select(1,65,79:88)

Schedule = hoopR::nba_schedule(league_id = '00', season = year)

NBA24Filt <- NBA24Filt %>%
  left_join(Schedule %>% mutate(game_id = as.integer(game_id)) %>% select(game_id, game_date), by = "game_id")

NBA24Filt = NBA24Filt[NBA24Filt$D5 != 0,]

NBA24Filt$Season = year+1

return(NBA24Filt)
}






ProcessTOs <- function(file_name, year) { 
NBA24 = read.csv(file_name)
NBA24 <- NBA24 %>%
  mutate(event_type = case_when(
    event_type == 1 ~ "MAKE",
    event_type == 2 ~ "MISS",
    event_type == 3 ~ "FreeThrow",
    event_type == 4 ~ "Rebound",
    event_type == 5 ~ "Turnover",
    event_type == 6 ~ "Foul",
    event_type == 7 ~ "Violation",
    event_type == 8 ~ "Substitution",
    event_type == 9 ~ "Timeout",
    event_type == 10 ~ "JumpBall",
    event_type == 11 ~ "Ejection",
    event_type == 12 ~ "StartOfPeriod",
    event_type == 13 ~ "EndOfPeriod",
    event_type == 14 ~ "Empty",
    TRUE ~ as.character(event_type)  # Default case if none of the above
  ))



# Replace NA values with empty strings in the entire dataframe
NBA24 <- NBA24 %>%
  mutate(across(where(is.character), ~replace_na(., "")))


NBA24 <- NBA24  %>%
  mutate(
    TurnoverCounter = case_when(
      grepl("Turnover", event_type, ignore.case = TRUE) ~ 1,
      TRUE ~ 0
    ),
    Home_Action_Score = case_when(
      grepl("Free Throw", home_description, ignore.case = TRUE) & !grepl("MISS", home_description, ignore.case = TRUE) ~ 1,
      grepl("PTS", home_description, ignore.case = TRUE) & !grepl("3PT", home_description, ignore.case = TRUE) & event_type == "MAKE" ~ 2,
      grepl("3PT", home_description, ignore.case = TRUE) & !grepl("MISS", home_description, ignore.case = TRUE) ~ 3,
      TRUE ~ 0
    ),
    Away_Action_Score = case_when(
      grepl("Free Throw", visitor_description, ignore.case = TRUE) & !grepl("MISS", visitor_description, ignore.case = TRUE) ~ 1,
      grepl("PTS", visitor_description, ignore.case = TRUE) & !grepl("3PT", visitor_description, ignore.case = TRUE) & event_type == "MAKE" ~ 2,
      grepl("3PT", visitor_description, ignore.case = TRUE) & !grepl("MISS", visitor_description, ignore.case = TRUE) ~ 3,
      TRUE ~ 0
    ),
    Net_Home = cumsum(Home_Action_Score),
    Net_Away = cumsum(Away_Action_Score),
    Net_Total = Net_Home + Net_Away,
    TurnoverTotal = cumsum(TurnoverCounter)
  )
rstudioapi::writeRStudioPreference("data_viewer_max_columns", 1000L)


NBA24 <- NBA24 %>%
  mutate(
    prev_seconds = lag(seconds_remaining_quarter, 1),
    Prev_Event = lag(event_type, 1, default = ""),
    Prev_visitor_desc = lag(visitor_description, 1, default = ""),  # visitor description from 1 row back
    Prev_visitor_desc2 = lag(visitor_description, 2, default = ""),
    Prev_home_desc = lag(home_description, 1, default = ""),  # home description from 1 row back
    Prev_home_desc2 = lag(home_description, 2, default = ""),
    Next_home_desc = lead(home_description, default = ""),  # Shift home descriptions down
    Next_visitor_desc = lead(visitor_description, default = ""),  # Shift visitor descriptions down
    offensive_FT_Rebound = case_when(
      str_detect(visitor_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") & str_detect(Prev_visitor_desc, "Free Throw")  ~ TRUE,
      str_detect(visitor_description, "rebound") & str_detect(Prev_visitor_desc, "MISS") & str_detect(Prev_visitor_desc, "Free Throw") ~ TRUE,
      str_detect(home_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") & str_detect(Prev_home_desc, "Free Throw") ~ TRUE,
      str_detect(home_description, "rebound") & str_detect(Prev_home_desc, "MISS") & str_detect(Prev_home_desc, "Free Throw") ~ TRUE,
      TRUE ~ FALSE
    ),
    End_of_Possession = case_when(
      str_detect(event_type, "EndOfPeriod") & prev_seconds > 0 ~ TRUE,
      str_detect(home_description, "Flagrant 2 of 2") | str_detect(home_description, "Flagrant 3 of 3") ~ TRUE,
      str_detect(visitor_description, "Flagrant 2 of 2") | str_detect(visitor_description, "Flagrant 3 of 3") ~ TRUE,
      str_detect(event_type, "Turnover") ~ TRUE,
      str_detect(home_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") & (!str_detect(Prev_Event, "FreeThrow")) ~ TRUE,
      str_detect(home_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS") & (!str_detect(Prev_Event, "FreeThrow"))~ TRUE,
      str_detect(visitor_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") & (!str_detect(Prev_Event, "FreeThrow")) ~ TRUE,
      str_detect(visitor_description, "Rebound") & str_detect(Prev_home_desc, "MISS") & (!str_detect(Prev_Event, "FreeThrow")) ~ TRUE,
      str_detect(home_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_visitor_desc, "S.FOUL") ~ TRUE,
      str_detect(visitor_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_home_desc, "S.FOUL") ~ TRUE,
      (str_detect(home_description, "1 of 1") | str_detect(home_description, "2 of 2") | str_detect(home_description, "3 of 3")) & 
        (!str_detect(Prev_home_desc2, "Transition") & !str_detect(Prev_home_desc, "Transition") & !str_detect(visitor_description, "Flagrant")) ~  TRUE,
      (str_detect(visitor_description, "1 of 1") | str_detect(visitor_description, "2 of 2") | str_detect(visitor_description, "3 of 3")) & 
        (!str_detect(Prev_home_desc2, "Transition") & !str_detect(Prev_home_desc, "Transition") & !str_detect(visitor_description, "Flagrant")) ~  TRUE,
      TRUE ~ FALSE  # Default case to FALSE
    ),     
    TeamOnOffense = case_when(
      (str_detect(home_description, "Flagrant 2 of 2") | str_detect(home_description, "Flagrant 3 of 3"))  ~ "Home",
      (str_detect(visitor_description, "Flagrant 2 of 2") | str_detect(visitor_description, "Flagrant 3 of 3"))  ~ "Away",
      str_detect(home_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS")  ~ "Away",
      str_detect(home_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS")  ~ "Away",
      str_detect(visitor_description, "REBOUND") & str_detect(Prev_home_desc, "MISS")   ~ "Home",
      str_detect(visitor_description, "Rebound") & str_detect(Prev_home_desc, "MISS")  ~ "Home",
      str_detect(home_description, "Turnover")  ~ "Home",
      str_detect(visitor_description, "Turnover")   ~ "Away",
      str_detect(home_description, "PTS")  ~ "Home",
      str_detect(visitor_description, "PTS")  ~ "Away",
      str_detect(home_description, "MISS")  ~ "Home",
      str_detect(visitor_description, "MISS")  ~ "Away",
      str_detect(home_description, "1 of 1") | str_detect(home_description, "2 of 2") | str_detect(home_description, "3 of 3") ~ "Home",
      str_detect(visitor_description, "1 of 1") | str_detect(visitor_description, "2 of 2") | str_detect(visitor_description, "3 of 3") ~ "Away",
      TRUE ~ ""  # Default case to NA
    )
  ) 

NBA24 <- NBA24 %>%
  mutate(
    FTEvents = rollapply(event_type == "FreeThrow", width = 5, FUN = sum, align = "left", fill = 0),
    SubEvents = rollapply(event_type == "Substitution", width = 5, FUN = sum, align = "left", fill = 0),
    PotentialFoul = case_when(
      (FTEvents != 0) & (SubEvents != 0) &
        str_detect(event_type, "Foul") & 
        !str_detect(visitor_description, "T.FOUL") & 
        !str_detect(visitor_description, "FLAGRANT") & 
        !str_detect(visitor_description, "Offens") & 
        !str_detect(visitor_description, "Transition") & 
        !str_detect(home_description, "T.FOUL") & 
        !str_detect(home_description, "Transition") & 
        !str_detect(home_description, "FLAGRANT") ~ TRUE,
      TRUE ~ FALSE
    ),
  ) %>%
  select(-FTEvents,-SubEvents) 

sum(NBA24$PotentialFoul)

# Show updated data frame


# Assuming your data frame is named df
NBA24 <- NBA24 %>%
  mutate(
    a1 = away_player1,
    a2 = away_player2,
    a3 = away_player3,
    a4 = away_player4,
    a5 = away_player5,
    h1 = home_player1,
    h2 = home_player2,
    h3 = home_player3,
    h4 = home_player4,
    h5 = home_player5
  )



foul_rows <- which(NBA24$PotentialFoul == TRUE)
foul_rows <- c(foul_rows, nrow(NBA24)) 
foul_rows[1]

propagate_player_values <- function(df,a) {
  for (i in 1:nrow(df)) {
    i = foul_rows[a]
    print(i)
    if (df$PotentialFoul[i] == TRUE) {
      # Store the a1-a5 and h1-h5 values from the current row
      a_values <- df[i, c("a1", "a2", "a3", "a4", "a5")]
      h_values <- df[i, c("h1", "h2", "h3", "h4", "h5")]
      
      # Initialize a flag to check for free throw
      found_free_throw <- FALSE
      
      # Loop through the rows below until the next End_of_Possession
      for (j in (i+1):nrow(df)) {
        
        if (df$event_type[j] == "Foul") {
          a = a + 1
          break
        }
        # If a free throw is found, set the flag
        if (df$event_type[j] == "FreeThrow") {
          found_free_throw <- TRUE
        }
        
        # Check for the next End_of_Possession row
        if (df$End_of_Possession[j] == TRUE) {
          # If free throw was found, replace the values from i+1 to j-1
          if (found_free_throw) {
            for (k in (i+1):(j)) {
              df[k, c("a1", "a2", "a3", "a4", "a5")] <- a_values
              df[k, c("h1", "h2", "h3", "h4", "h5")] <- h_values
            }
          }
          a = a + 1
          break
        }
        if (df$offensive_FT_Rebound[j] == TRUE) {
          if (found_free_throw) {
            for (k in (i+1):(j-1)) {
              df[k, c("a1", "a2", "a3", "a4", "a5")] <- a_values
              df[k, c("h1", "h2", "h3", "h4", "h5")] <- h_values
            }
          }
          a = a + 1
          break
        }
      }
    }
  }
  return(df)
}

FTOffCheck <- function(df,a) {
  for (i in 1:nrow(df)) {
    i = foul_rows[a]
    print(i)
    if (df$PotentialFoul[i] == TRUE) {
      # Store the a1-a5 and h1-h5 values from the current row
      values <- df[i, c("a1", "a2", "a3", "a4", "a5", "h1", "h2", "h3", "h4", "h5")]
      
      # Loop through the rows ielow until the next End_of_Possession
      for (j in (i+1):nrow(df)) {
        # If a free throw is found, set the flag
        if (df$offensive_FT_Rebound[j] == TRUE) {
          if (any(df[j, c("a1", "a2", "a3", "a4", "a5", "h1", "h2", "h3", "h4", "h5")] != values)) {
            print("Hello")
            df$End_of_Possession[j-1] = TRUE
            a = a + 1
            break
          }
        }
        # Check for the next End_of_Possession row
        if (df$End_of_Possession[j] == TRUE) {
          a = a + 1
          break
        }
      }
    }
  }
  return(df)
}


# Assuming df is your dataframe

NBA24Check <- propagate_player_values(NBA24,1)
NBA24Check  <- FTOffCheck(NBA24,1)

NBA24 <- NBA24 %>%
  mutate(
    O1 = case_when(
      TeamOnOffense == "Away" ~ a1,
      TeamOnOffense == "Home" ~ h1,
      TRUE ~ 0
    ),
    O2 = case_when(
      TeamOnOffense == "Away" ~ a2,
      TeamOnOffense == "Home" ~ h2,
      TRUE ~ 0
    ),
    O3 = case_when(
      TeamOnOffense == "Away" ~ a3,
      TeamOnOffense == "Home" ~ h3,
      TRUE ~ 0
    ),
    O4 = case_when(
      TeamOnOffense == "Away" ~ a4,
      TeamOnOffense == "Home" ~ h4,
      TRUE ~ 0
    ),
    O5 = case_when(
      TeamOnOffense == "Away" ~ a5,
      TeamOnOffense == "Home" ~ h5,
      TRUE ~ 0
    ),
    D1 = case_when(
      TeamOnOffense == "Away" ~ h1,
      TeamOnOffense == "Home" ~ a1,
      TRUE ~ 0
    ),
    D2 = case_when(
      TeamOnOffense == "Away" ~ h2,
      TeamOnOffense == "Home" ~ a2,
      TRUE ~ 0
    ),
    D3 = case_when(
      TeamOnOffense == "Away" ~ h3,
      TeamOnOffense == "Home" ~ a3,
      TRUE ~ 0
    ),
    D4 = case_when(
      TeamOnOffense == "Away" ~ h4,
      TeamOnOffense == "Home" ~ a4,
      TRUE ~ 0
    ),
    D5 = case_when(
      TeamOnOffense == "Away" ~ h5,
      TeamOnOffense == "Home" ~ a5,
      TRUE ~ 0
    )
  )



progressr::with_progress({
  Player24 <- hoopR::nba_commonallplayers(league_id = '00', season = 2024)
})

Players = as.data.frame(Player24)


Dict24 <- Players[, c(1, 3)]


NBA24Filt <- NBA24[NBA24$End_of_Possession == TRUE, ]


for (i in 1:nrow(Dict24)) {
  value_to_replace <- Dict24[i,1]
  replacement_value <- Dict24[i,2]
  
  NBA24Filt[, 81:90] <- lapply(NBA24Filt[, 81:90], function(x) {
    replace(x, x == value_to_replace, replacement_value)
  })
}


NBA24Filt  <- NBA24Filt  %>%
  select(1,52,81:90)

Schedule = hoopR::nba_schedule(league_id = '00', season = year)

NBA24Filt <- NBA24Filt %>%
  left_join(Schedule %>% mutate(game_id = as.integer(game_id)) %>% select(game_id, game_date), by = "game_id")

NBA24Filt = NBA24Filt[NBA24Filt$D5 != 0,]

NBA24Filt$Season = year+1

return(NBA24Filt)

}








ProcessTS <- function(file_name, year) {
NBA24 = read.csv(file_name)
NBA24 <- NBA24 %>%
  mutate(event_type = case_when(
    event_type == 1 ~ "MAKE",
    event_type == 2 ~ "MISS",
    event_type == 3 ~ "FreeThrow",
    event_type == 4 ~ "Rebound",
    event_type == 5 ~ "Turnover",
    event_type == 6 ~ "Foul",
    event_type == 7 ~ "Violation",
    event_type == 8 ~ "Substitution",
    event_type == 9 ~ "Timeout",
    event_type == 10 ~ "JumpBall",
    event_type == 11 ~ "Ejection",
    event_type == 12 ~ "StartOfPeriod",
    event_type == 13 ~ "EndOfPeriod",
    event_type == 14 ~ "Empty",
    TRUE ~ as.character(event_type)  # Default case if none of the above
  ))

library(dplyr)
library(stringr)
library(tidyr)

# Replace NA values with empty strings in the entire dataframe
NBA24 <- NBA24 %>%
  mutate(across(where(is.character), ~replace_na(., "")))


NBA24 <- NBA24  %>%
  mutate(
    Home_Action_Score = case_when(
      grepl("Free Throw", home_description, ignore.case = TRUE) & !grepl("MISS", home_description, ignore.case = TRUE) ~ 1,
      grepl("PTS", home_description, ignore.case = TRUE) & !grepl("3PT", home_description, ignore.case = TRUE) & event_type == "MAKE" ~ 2,
      grepl("3PT", home_description, ignore.case = TRUE) & !grepl("MISS", home_description, ignore.case = TRUE) ~ 3,
      TRUE ~ 0
    ),
    Away_Action_Score = case_when(
      grepl("Free Throw", visitor_description, ignore.case = TRUE) & !grepl("MISS", visitor_description, ignore.case = TRUE) ~ 1,
      grepl("PTS", visitor_description, ignore.case = TRUE) & !grepl("3PT", visitor_description, ignore.case = TRUE) & event_type == "MAKE" ~ 2,
      grepl("3PT", visitor_description, ignore.case = TRUE) & !grepl("MISS", visitor_description, ignore.case = TRUE) ~ 3,
      TRUE ~ 0
    ),
    Net_Home = cumsum(Home_Action_Score),
    Net_Away = cumsum(Away_Action_Score),
    Net_Total = Net_Home + Net_Away
  )
rstudioapi::writeRStudioPreference("data_viewer_max_columns", 1000L)


NBA24 <- NBA24 %>%
  mutate(
    prev_seconds = lag(seconds_remaining_quarter, 1),
    Prev_visitor_desc = lag(visitor_description, 1, default = ""),  # visitor description from 1 row back
    Prev_visitor_desc2 = lag(visitor_description, 2, default = ""),
    Prev_home_desc = lag(home_description, 1, default = ""),  # home description from 1 row back
    Prev_home_desc2 = lag(home_description, 2, default = ""),
    Next_home_desc = lead(home_description, default = ""),  # Shift home descriptions down
    Next_visitor_desc = lead(visitor_description, default = ""),  # Shift visitor descriptions down
    End_of_Possession = case_when(
      str_detect(event_type, "EndOfPeriod") & prev_seconds > 0 ~ TRUE,
      str_detect(home_description, "Flagrant 2 of 2") | str_detect(home_description, "Flagrant 3 of 3") ~ TRUE,
      str_detect(visitor_description, "Flagrant 2 of 2") | str_detect(visitor_description, "Flagrant 3 of 3") ~ TRUE,
      str_detect(event_type, "Turnover") ~ TRUE,
      str_detect(visitor_description, "MISS") & str_detect(event_type, "MISS") ~ TRUE,
      str_detect(home_description, "MISS") & str_detect(event_type, "MISS") ~ TRUE,
      str_detect(home_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_visitor_desc, "S.FOUL") ~ TRUE,
      str_detect(visitor_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_home_desc, "S.FOUL") ~ TRUE,
      (str_detect(home_description, "1 of 1") | str_detect(home_description, "2 of 2") | str_detect(home_description, "3 of 3"))  & 
        (!str_detect(Prev_home_desc2, "Transition") & !str_detect(Prev_home_desc, "Transition") & !str_detect(visitor_description, "Flagrant")) ~  TRUE,
      (str_detect(visitor_description, "1 of 1") | str_detect(visitor_description, "2 of 2") | str_detect(visitor_description, "3 of 3"))  & 
        (!str_detect(Prev_home_desc2, "Transition") & !str_detect(Prev_home_desc, "Transition") & !str_detect(visitor_description, "Flagrant")) ~  TRUE,
      TRUE ~ FALSE  # Default case to FALSE
    ),     
    TeamOnOffense = case_when(
      (str_detect(home_description, "Flagrant 2 of 2") | str_detect(home_description, "Flagrant 3 of 3"))  ~ "Home",
      (str_detect(visitor_description, "Flagrant 2 of 2") | str_detect(visitor_description, "Flagrant 3 of 3"))  ~ "Away",
      str_detect(home_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS")  ~ "Away",
      str_detect(home_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS")  ~ "Away",
      str_detect(visitor_description, "REBOUND") & str_detect(Prev_home_desc, "MISS")  ~ "Home",
      str_detect(visitor_description, "Rebound") & str_detect(Prev_home_desc, "MISS")  ~ "Home",
      str_detect(visitor_description, "MISS") ~ "Away",
      str_detect(home_description, "MISS")  ~ "Home",
      str_detect(visitor_description, "MAKE") ~ "Away",
      str_detect(home_description, "MAKE")  ~ "Home",
      str_detect(home_description, "Turnover")  ~ "Home",
      str_detect(visitor_description, "Turnover")   ~ "Away",
      str_detect(home_description, "PTS")  ~ "Home",
      str_detect(visitor_description, "PTS") ~ "Away",
      (str_detect(home_description, "1 of 1") | str_detect(home_description, "2 of 2") | str_detect(home_description, "3 of 3")) ~ "Home",
      (str_detect(visitor_description, "1 of 1") | str_detect(visitor_description, "2 of 2") | str_detect(visitor_description, "3 of 3")) ~ "Away",
      TRUE ~ ""  # Default case to NA
    )
  ) 

NBA24 <- NBA24 %>%
  mutate(
    FTEvents = rollapply(event_type == "FreeThrow", width = 5, FUN = sum, align = "left", fill = 0),
    SubEvents = rollapply(event_type == "Substitution", width = 5, FUN = sum, align = "left", fill = 0),
    PotentialFoul = case_when(
      (FTEvents != 0) & (SubEvents != 0) &
        str_detect(event_type, "Foul") & 
        !str_detect(visitor_description, "T.FOUL") & 
        !str_detect(visitor_description, "FLAGRANT") & 
        !str_detect(visitor_description, "Offens") & 
        !str_detect(visitor_description, "Transition") & 
        !str_detect(home_description, "T.FOUL") & 
        !str_detect(home_description, "Transition") & 
        !str_detect(home_description, "FLAGRANT") ~ TRUE,
      TRUE ~ FALSE
    ),
  ) %>%
  select(-FTEvents,-SubEvents) 

sum(NBA24$PotentialFoul)

# Show updated data frame


# Assuming your data frame is named df
NBA24 <- NBA24 %>%
  mutate(
    a1 = away_player1,
    a2 = away_player2,
    a3 = away_player3,
    a4 = away_player4,
    a5 = away_player5,
    h1 = home_player1,
    h2 = home_player2,
    h3 = home_player3,
    h4 = home_player4,
    h5 = home_player5
  )



foul_rows <- which(NBA24$PotentialFoul == TRUE)
foul_rows <- c(foul_rows, nrow(NBA24)) 
foul_rows[1]

propagate_player_values <- function(df,a) {
  for (i in 1:nrow(df)) {
    i = foul_rows[a]
    print(i)
    if (df$PotentialFoul[i] == TRUE) {
      # Store the a1-a5 and h1-h5 values from the current row
      a_values <- df[i, c("a1", "a2", "a3", "a4", "a5")]
      h_values <- df[i, c("h1", "h2", "h3", "h4", "h5")]
      
      # Initialize a flag to check for free throw
      found_free_throw <- FALSE
      
      # Loop through the rows below until the next End_of_Possession
      for (j in (i+1):nrow(df)) {
        
        if (df$event_type[j] == "Foul") {
          a = a + 1
          break
        }
        # If a free throw is found, set the flag
        if (df$event_type[j] == "FreeThrow") {
          found_free_throw <- TRUE
        }
        
        # Check for the next End_of_Possession row
        if (df$End_of_Possession[j] == TRUE) {
          # If free throw was found, replace the values from i+1 to j-1
          if (found_free_throw) {
            for (k in (i+1):(j)) {
              df[k, c("a1", "a2", "a3", "a4", "a5")] <- a_values
              df[k, c("h1", "h2", "h3", "h4", "h5")] <- h_values
            }
          }
          a = a + 1
          break
        }
        if (df$offensive_FT_Rebound[j] == TRUE) {
          if (found_free_throw) {
            for (k in (i+1):(j-1)) {
              df[k, c("a1", "a2", "a3", "a4", "a5")] <- a_values
              df[k, c("h1", "h2", "h3", "h4", "h5")] <- h_values
            }
          }
          a = a + 1
          break
        }
      }
    }
  }
  return(df)
}




# Assuming df is your dataframe

NBA24Check <- propagate_player_values(NBA24,1)


NBA24 <- NBA24 %>%
  mutate(
    O1 = case_when(
      TeamOnOffense == "Away" ~ a1,
      TeamOnOffense == "Home" ~ h1,
      TRUE ~ 0
    ),
    O2 = case_when(
      TeamOnOffense == "Away" ~ a2,
      TeamOnOffense == "Home" ~ h2,
      TRUE ~ 0
    ),
    O3 = case_when(
      TeamOnOffense == "Away" ~ a3,
      TeamOnOffense == "Home" ~ h3,
      TRUE ~ 0
    ),
    O4 = case_when(
      TeamOnOffense == "Away" ~ a4,
      TeamOnOffense == "Home" ~ h4,
      TRUE ~ 0
    ),
    O5 = case_when(
      TeamOnOffense == "Away" ~ a5,
      TeamOnOffense == "Home" ~ h5,
      TRUE ~ 0
    ),
    D1 = case_when(
      TeamOnOffense == "Away" ~ h1,
      TeamOnOffense == "Home" ~ a1,
      TRUE ~ 0
    ),
    D2 = case_when(
      TeamOnOffense == "Away" ~ h2,
      TeamOnOffense == "Home" ~ a2,
      TRUE ~ 0
    ),
    D3 = case_when(
      TeamOnOffense == "Away" ~ h3,
      TeamOnOffense == "Home" ~ a3,
      TRUE ~ 0
    ),
    D4 = case_when(
      TeamOnOffense == "Away" ~ h4,
      TeamOnOffense == "Home" ~ a4,
      TRUE ~ 0
    ),
    D5 = case_when(
      TeamOnOffense == "Away" ~ h5,
      TeamOnOffense == "Home" ~ a5,
      TRUE ~ 0
    )
  )



progressr::with_progress({
  Player24 <- hoopR::nba_commonallplayers(league_id = '00', season = 2024)
})

Players = as.data.frame(Player24)


Dict24 <- Players[, c(1, 3)]


NBA24Filt <- NBA24[NBA24$End_of_Possession == TRUE, ]

NBA24Filt = NBA24Filt[NBA24Filt$event_type != "Turnover", ]


for (i in 1:nrow(Dict24)) {
  value_to_replace <- Dict24[i,1]
  replacement_value <- Dict24[i,2]
  
  NBA24Filt[, 77:86] <- lapply(NBA24Filt[, 77:86], function(x) {
    replace(x, x == value_to_replace, replacement_value)
  })
}

NBA24Filt <- NBA24Filt %>%
  mutate(Net_Diff = Net_Total - lag(Net_Total, default = 0))

NBA24Filt  <- NBA24Filt  %>%
  select(1,77:87)

Schedule = hoopR::nba_schedule(league_id = '00', season = year)

NBA24Filt <- NBA24Filt %>%
  left_join(Schedule %>% mutate(game_id = as.integer(game_id)) %>% select(game_id, game_date), by = "game_id")

NBA24Filt = NBA24Filt[NBA24Filt$D5 != 0,]

NBA24Filt$Season = year+1

return(NBA24Filt)
}
























# For 2024 data (NBA24.csv)
Rebounds24 = ProcessRebounding("NBA24.csv", 2023)
RAPM24 = ProcessRAPM("NBA24.csv", 2023, "2023-24")
TO24 = ProcessTOs("NBA24.csv", 2023)
TS24 = ProcessTS("NBA24.csv", 2023)

write.csv(Rebounds24, "Rebounds24.csv", row.names = FALSE)
write.csv(RAPM24, "RAPM24.csv", row.names = FALSE)
write.csv(TO24, "TO24.csv", row.names = FALSE)
write.csv(TS24, "TS24.csv", row.names = FALSE)

# For 2023 data (NBA23.csv)
Rebounds23 = ProcessRebounding("NBA23.csv", 2022)
RAPM23 = ProcessRAPM("NBA23.csv", 2022, "2022-23")
TO23 = ProcessTOs("NBA23.csv", 2022)
TS23 = ProcessTS("NBA23.csv", 2022)

write.csv(Rebounds23, "Rebounds23.csv", row.names = FALSE)
write.csv(RAPM23, "RAPM23.csv", row.names = FALSE)
write.csv(TO23, "TO23.csv", row.names = FALSE)
write.csv(TS23, "TS23.csv", row.names = FALSE)

# For 2022 data (NBA22.csv)
Rebounds22 = ProcessRebounding("NBA22.csv", 2021)
RAPM22 = ProcessRAPM("NBA22.csv", 2021, "2021-22")
TO22 = ProcessTOs("NBA22.csv", 2021)
TS22 = ProcessTS("NBA22.csv", 2021)

write.csv(Rebounds22, "Rebounds22.csv", row.names = FALSE)
write.csv(RAPM22, "RAPM22.csv", row.names = FALSE)
write.csv(TO22, "TO22.csv", row.names = FALSE)
write.csv(TS22, "TS22.csv", row.names = FALSE)

# For 2021 data (NBA21.csv)
Rebounds21 = ProcessRebounding("NBA21.csv", 2020)
RAPM21 = ProcessRAPM("NBA21.csv", 2020, "2020-21")
TO21 = ProcessTOs("NBA21.csv", 2020)
TS21 = ProcessTS("NBA21.csv", 2020)

write.csv(Rebounds21, "Rebounds21.csv", row.names = FALSE)
write.csv(RAPM21, "RAPM21.csv", row.names = FALSE)
write.csv(TO21, "TO21.csv", row.names = FALSE)
write.csv(TS21, "TS21.csv", row.names = FALSE)

# For 2020 data (NBA20.csv)
Rebounds20 = ProcessRebounding("NBA20.csv", 2019)
RAPM20 = ProcessRAPM("NBA20.csv", 2019, "2019-20")
TO20 = ProcessTOs("NBA20.csv", 2019)
TS20 = ProcessTS("NBA20.csv", 2019)

write.csv(Rebounds20, "Rebounds20.csv", row.names = FALSE)
write.csv(RAPM20, "RAPM20.csv", row.names = FALSE)
write.csv(TO20, "TO20.csv", row.names = FALSE)
write.csv(TS20, "TS20.csv", row.names = FALSE)

# For 2019 data (NBA19.csv)
Rebounds19 = ProcessRebounding("NBA19.csv", 2018)
RAPM19 = ProcessRAPM("NBA19.csv", 2018, "2018-19")
TO19 = ProcessTOs("NBA19.csv", 2018)
TS19 = ProcessTS("NBA19.csv", 2018)

write.csv(Rebounds19, "Rebounds19.csv", row.names = FALSE)
write.csv(RAPM19, "RAPM19.csv", row.names = FALSE)
write.csv(TO19, "TO19.csv", row.names = FALSE)
write.csv(TS19, "TS19.csv", row.names = FALSE)

# For 2018 data (NBA18.csv)
Rebounds18 = ProcessRebounding("NBA18.csv", 2017)
RAPM18 = ProcessRAPM("NBA18.csv", 2017, "2017-18")
TO18 = ProcessTOs("NBA18.csv", 2017)
TS18 = ProcessTS("NBA18.csv", 2017)

write.csv(Rebounds18, "Rebounds18.csv", row.names = FALSE)
write.csv(RAPM18, "RAPM18.csv", row.names = FALSE)
write.csv(TO18, "TO18.csv", row.names = FALSE)
write.csv(TS18, "TS18.csv", row.names = FALSE)

# For 2017 data (NBA17.csv)
Rebounds17 = ProcessRebounding("NBA17.csv", 2016)
RAPM17 = ProcessRAPM("NBA17.csv", 2016, "2016-17")
TO17 = ProcessTOs("NBA17.csv", 2016)
TS17 = ProcessTS("NBA17.csv", 2016)

write.csv(Rebounds17, "Rebounds17.csv", row.names = FALSE)
write.csv(RAPM17, "RAPM17.csv", row.names = FALSE)
write.csv(TO17, "TO17.csv", row.names = FALSE)
write.csv(TS17, "TS17.csv", row.names = FALSE)

# For 2016 data (NBA16.csv)
Rebounds16 = ProcessRebounding("NBA16.csv", 2015)
RAPM16 = ProcessRAPM("NBA16.csv", 2015, "2015-16")
TO16 = ProcessTOs("NBA16.csv", 2015)
TS16 = ProcessTS("NBA16.csv", 2015)

write.csv(Rebounds16, "Rebounds16.csv", row.names = FALSE)
write.csv(RAPM16, "RAPM16.csv", row.names = FALSE)
write.csv(TO16, "TO16.csv", row.names = FALSE)
write.csv(TS16, "TS16.csv", row.names = FALSE)

# For 2015 data (NBA15.csv)
Rebounds15 = ProcessRebounding("NBA15.csv", 2014)
RAPM15 = ProcessRAPM("NBA15.csv", 2014, "2014-15")
TO15 = ProcessTOs("NBA15.csv", 2014)
TS15 = ProcessTS("NBA15.csv", 2014)

write.csv(Rebounds15, "Rebounds15.csv", row.names = FALSE)
write.csv(RAPM15, "RAPM15.csv", row.names = FALSE)
write.csv(TO15, "TO15.csv", row.names = FALSE)
write.csv(TS15, "TS15.csv", row.names = FALSE)

# For 2014 data (NBA14.csv)
Rebounds14 = ProcessRebounding("NBA14.csv", 2013)
RAPM14 = ProcessRAPM("NBA14.csv", 2013, "2013-14")
TO14 = ProcessTOs("NBA14.csv", 2013)
TS14 = ProcessTS("NBA14.csv", 2013)

write.csv(Rebounds14, "Rebounds14.csv", row.names = FALSE)
write.csv(RAPM14, "RAPM14.csv", row.names = FALSE)
write.csv(TO14, "TO14.csv", row.names = FALSE)
write.csv(TS14, "TS14.csv", row.names = FALSE)

# For 2013 data (NBA13.csv)
Rebounds13 = ProcessRebounding("NBA13.csv", 2012)
RAPM13 = ProcessRAPM("NBA13.csv", 2012, "2012-13")
TO13 = ProcessTOs("NBA13.csv", 2012)
TS13 = ProcessTS("NBA13.csv", 2012)

write.csv(Rebounds13, "Rebounds13.csv", row.names = FALSE)
write.csv(RAPM13, "RAPM13.csv", row.names = FALSE)
write.csv(TO13, "TO13.csv", row.names = FALSE)
write.csv(TS13, "TS13.csv", row.names = FALSE)

# For 2012 data (NBA12.csv)
Rebounds12 = ProcessRebounding("NBA12.csv", 2011)
RAPM12 = ProcessRAPM("NBA12.csv", 2011, "2011-12")
TO12 = ProcessTOs("NBA12.csv", 2011)
TS12 = ProcessTS("NBA12.csv", 2011)

write.csv(Rebounds12, "Rebounds12.csv", row.names = FALSE)
write.csv(RAPM12, "RAPM12.csv", row.names = FALSE)
write.csv(TO12, "TO12.csv", row.names = FALSE)
write.csv(TS12, "TS12.csv", row.names = FALSE)

write.csv(Rebounds11, "Rebounds11.csv", row.names = FALSE)
write.csv(RAPM11, "RAPM11.csv", row.names = FALSE)
write.csv(TO11, "TO11.csv", row.names = FALSE)
write.csv(TS11, "TS11.csv", row.names = FALSE)

# For 2010 data (NBA10.csv)
Rebounds10 = ProcessRebounding("NBA10.csv", 2009)
RAPM10 = ProcessRAPM("NBA10.csv", 2009, "2009-10")
TO10 = ProcessTOs("NBA10.csv", 2009)
TS10 = ProcessTS("NBA10.csv", 2009)

write.csv(Rebounds10, "Rebounds10.csv", row.names = FALSE)
write.csv(RAPM10, "RAPM10.csv", row.names = FALSE)
write.csv(TO10, "TO10.csv", row.names = FALSE)
write.csv(TS10, "TS10.csv", row.names = FALSE)

# For 2009 data (NBA09.csv)
Rebounds09 = ProcessRebounding("NBA09.csv", 2008)
RAPM09 = ProcessRAPM("NBA09.csv", 2008, "2008-09")
TO09 = ProcessTOs("NBA09.csv", 2008)
TS09 = ProcessTS("NBA09.csv", 2008)

write.csv(Rebounds09, "Rebounds09.csv", row.names = FALSE)
write.csv(RAPM09, "RAPM09.csv", row.names = FALSE)
write.csv(TO09, "TO09.csv", row.names = FALSE)
write.csv(TS09, "TS09.csv", row.names = FALSE)

# For 2008 data (NBA08.csv)
Rebounds08 = ProcessRebounding("NBA08.csv", 2007)
RAPM08 = ProcessRAPM("NBA08.csv", 2007, "2007-08")
TO08 = ProcessTOs("NBA08.csv", 2007)
TS08 = ProcessTS("NBA08.csv", 2007)

write.csv(Rebounds08, "Rebounds08.csv", row.names = FALSE)
write.csv(RAPM08, "RAPM08.csv", row.names = FALSE)
write.csv(TO08, "TO08.csv", row.names = FALSE)
write.csv(TS08, "TS08.csv", row.names = FALSE)

Rebounds07 = ProcessRebounding("NBA07.csv", 2006)
RAPM07 = ProcessRAPM("NBA07.csv", 2006, "2006-07")
TO07 = ProcessTOs("NBA07.csv", 2006)
TS07 = ProcessTS("NBA07.csv", 2006)

write.csv(Rebounds07, "Rebounds07.csv", row.names = FALSE)
write.csv(RAPM07, "RAPM07.csv", row.names = FALSE)
write.csv(TO07, "TO07.csv", row.names = FALSE)
write.csv(TS07, "TS07.csv", row.names = FALSE)

# For 2006 data (NBA06.csv)
Rebounds06 = ProcessRebounding("NBA06.csv", 2005)
RAPM06 = ProcessRAPM("NBA06.csv", 2005, "2005-06")
TO06 = ProcessTOs("NBA06.csv", 2005)
TS06 = ProcessTS("NBA06.csv", 2005)

write.csv(Rebounds06, "Rebounds06.csv", row.names = FALSE)
write.csv(RAPM06, "RAPM06.csv", row.names = FALSE)
write.csv(TO06, "TO06.csv", row.names = FALSE)
write.csv(TS06, "TS06.csv", row.names = FALSE)

# For 2005 data (NBA05.csv)
Rebounds05 = ProcessRebounding("NBA05.csv", 2004)
RAPM05 = ProcessRAPM("NBA05.csv", 2004, "2004-05")
TO05 = ProcessTOs("NBA05.csv", 2004)
TS05 = ProcessTS("NBA05.csv", 2004)

write.csv(Rebounds05, "Rebounds05.csv", row.names = FALSE)
write.csv(RAPM05, "RAPM05.csv", row.names = FALSE)
write.csv(TO05, "TO05.csv", row.names = FALSE)
write.csv(TS05, "TS05.csv", row.names = FALSE)

# For 2004 data (NBA04.csv)
Rebounds04 = ProcessRebounding("NBA04.csv", 2003)
RAPM04 = ProcessRAPM("NBA04.csv", 2003, "2003-04")
TO04 = ProcessTOs("NBA04.csv", 2003)
TS04 = ProcessTS("NBA04.csv", 2003)

write.csv(Rebounds04, "Rebounds04.csv", row.names = FALSE)
write.csv(RAPM04, "RAPM04.csv", row.names = FALSE)
write.csv(TO04, "TO04.csv", row.names = FALSE)
write.csv(TS04, "TS04.csv", row.names = FALSE)

# For 2003 data (NBA03.csv)
Rebounds03 = ProcessRebounding("NBA03.csv", 2002)
RAPM03 = ProcessRAPM("NBA03.csv", 2002, "2002-03")
TO03 = ProcessTOs("NBA03.csv", 2002)
TS03 = ProcessTS("NBA03.csv", 2002)

write.csv(Rebounds03, "Rebounds03.csv", row.names = FALSE)
write.csv(RAPM03, "RAPM03.csv", row.names = FALSE)
write.csv(TO03, "TO03.csv", row.names = FALSE)
write.csv(TS03, "TS03.csv", row.names = FALSE)

# For 2002 data (NBA02.csv)
Rebounds02 = ProcessRebounding("NBA02.csv", 2001)
RAPM02 = ProcessRAPM("NBA02.csv", 2001, "2001-02")
TO02 = ProcessTOs("NBA02.csv", 2001)
TS02 = ProcessTS("NBA02.csv", 2001)

write.csv(Rebounds02, "Rebounds02.csv", row.names = FALSE)
write.csv(RAPM02, "RAPM02.csv", row.names = FALSE)
write.csv(TO02, "TO02.csv", row.names = FALSE)
write.csv(TS02, "TS02.csv", row.names = FALSE)

# For 2001 data (NBA01.csv)
Rebounds01 = ProcessRebounding("NBA01.csv", 2000)
RAPM01 = ProcessRAPM("NBA01.csv", 2000, "2000-01")
TO01 = ProcessTOs("NBA01.csv", 2000)
TS01 = ProcessTS("NBA01.csv", 2000)

write.csv(Rebounds01, "Rebounds01.csv", row.names = FALSE)
write.csv(RAPM01, "RAPM01.csv", row.names = FALSE)
write.csv(TO01, "TO01.csv", row.names = FALSE)
write.csv(TS01, "TS01.csv", row.names = FALSE)

Rebounds01 = ProcessRebounding("NBA01.csv", 2000)
RAPM01 = ProcessRAPM("NBA01.csv", 2000, "2000-01")
TO01 = ProcessTOs("NBA01.csv", 2000)
TS01 = ProcessTS("NBA01.csv", 2000)

write.csv(Rebounds01, "Rebounds01.csv", row.names = FALSE)
write.csv(TO01, "TO01.csv", row.names = FALSE)
write.csv(TS01, "TS01.csv", row.names = FALSE)

modify_dataframe <- function(df) {
  df <- df[-1]           # Remove the first column of the dataframe
  df$possessions <- 1    # Set all values in the possessions column to 1
  return(df)
}



turnovers_list = list(Turnovers00,Turnovers01,Turnovers02,Turnovers03,Turnovers04,Turnovers05,Turnovers06,Turnovers07,Turnovers08,Turnovers09,Turnovers10,Turnovers11,
     Turnovers12,Turnovers13,Turnovers14,Turnovers15,Turnovers16,Turnovers17,Turnovers18,Turnovers19,Turnovers20,Turnovers21,Turnovers22,Turnovers23,Turnovers24)

rebounds_list = list(Rebounds00,Rebounds01,Rebounds02,Rebounds03,Rebounds04,Rebounds05,Rebounds06,Rebounds07,Rebounds08,Rebounds09,Rebounds10,Rebounds11,Rebounds12,
     Rebounds13,Rebounds14,Rebounds15,Rebounds16,Rebounds17,Rebounds18,Rebounds19,Rebounds20,Rebounds21,Rebounds22,Rebounds23,Rebounds24)

ts_list = list(TS00,TS01,TS02,TS03,TS04,TS05,TS06,TS07,TS08,TS09,TS10,TS11,
     TS12,TS13,TS14,TS15,TS16,TS17,TS18,TS19,TS20,TS21,TS22,TS23,TS24)

RAPMlist = list(RAPM00,RAPM01,RAPM02,RAPM03,RAPM04,RAPM05,RAPM06,RAPM07,RAPM08,RAPM09,RAPM10,RAPM11,
     RAPM12,RAPM13,RAPM14,RAPM15,RAPM16,RAPM17,RAPM18,RAPM19,RAPM20,RAPM21,RAPM22,RAPM23,RAPM24)

# Function to modify each dataframe (remove the first column and add possessions column)


# Modify each dataframe in the list (for turnovers, rebounds, and TS)
modified_turnovers <- lapply(turnovers_list, modify_dataframe)
modified_rebounds <- lapply(rebounds_list, modify_dataframe)
modified_ts <- lapply(ts_list, modify_dataframe)
modified_rapm <- lapply(rapm_list, modify_dataframe)

# Combine all modified dataframes for each category
Combined_Turnovers <- do.call(rbind, modified_turnovers)
Combined_Rebounds <- do.call(rbind, modified_rebounds)
Combined_TS <- do.call(rbind, modified_ts)
Combined_Rapm<- do.call(rbind, modified_rapm)


install.packages("RMySQL")
library(RMySQL)

# Establish a connection to the MySQL database using RMySQL
con <- dbConnect(RMySQL::MySQL(),host = "127.0.0.1",             # Host (localhost in this case)
                 dbname = "my_new_database",     # Database name
                 user = "root",                  # Username
                 password = "tijaytijay",        # Password
                 port = 3306)                    # Default MySQL port

# Write the dataframe 'NBAPOSSTOTAL' to the SQL database as a new table named 'NBASTATS'
dbWriteTable(con, "TOS", Combined_Turnovers, overwrite = TRUE, row.names = FALSE)
dbWriteTable(con, "TOS",Combined_Rebounds, overwrite = TRUE, row.names = FALSE)
dbWriteTable(con, "TOS", Combined_TS, overwrite = TRUE, row.names = FALSE)
dbWriteTable(con, "TOS", Combined_Rapm, overwrite = TRUE, row.names = FALSE)

# Display the retrieved data
print(OffenseDataframe)

# Close the connection after the operation
dbDisconnect(con)





