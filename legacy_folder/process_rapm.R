library(dplyr)
library(stringr)
library(tidyr)
library(zoo)

ProcessRAPM <- function(file_name, year, Season) {
  
  
  NBA24 = read.csv(file_name) 
  NBA24 = NBA24 %>% filter(if_all(c("away_player1", "away_player2", "away_player3", "away_player4", "away_player5", "home_player1", "home_player2", "home_player3", "home_player4", "home_player5"), ~ !is.na(.) & . != ""))
  
  
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
  
  print("Hello")
  
  NBA24 <- NBA24 %>%
    mutate(
      prev_seconds = lag(seconds_remaining_quarter, 1),
      Prev_visitor_desc = lag(visitor_description, 1, default = ""),  # visitor description from 1 row back
      Prev_visitor_desc2 = lag(visitor_description, 2, default = ""),
      Prev_home_desc = lag(home_description, 1, default = ""),  # home description from 1 row back
      Prev_home_desc2 = lag(home_description, 2, default = ""),
      Next_home_desc = lead(home_description, default = ""),  # Shift home descriptions down
      Next_visitor_desc = lead(visitor_description, default = ""),  # Shift visitor descriptions down
      offensive_FT_Rebound = case_when(
        str_detect(visitor_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") & str_detect(Prev_visitor_desc, "Free Throw") & 
          (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))  ~ TRUE,
        str_detect(visitor_description, "rebound") & str_detect(Prev_visitor_desc, "MISS") & str_detect(Prev_visitor_desc, "Free Throw") & 
          (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(home_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") & str_detect(Prev_home_desc, "Free Throw") & 
          (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(home_description, "rebound") & str_detect(Prev_home_desc, "MISS") & str_detect(Prev_home_desc, "Free Throw") & 
          (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        TRUE ~ FALSE
      ),
      End_of_Possession = case_when(
        str_detect(event_type, "EndOfPeriod") & prev_seconds > 0 ~ TRUE,
        str_detect(home_description, "Flagrant 2 of 2") | str_detect(home_description, "Flagrant 3 of 3") ~ TRUE,
        str_detect(visitor_description, "Flagrant 2 of 2") | str_detect(visitor_description, "Flagrant 3 of 3") ~ TRUE,
        str_detect(event_type, "Turnover") ~ TRUE,
        str_detect(home_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") & 
          (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(home_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS") & 
          (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(visitor_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") & 
          (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(visitor_description, "Rebound") & str_detect(Prev_home_desc, "MISS") & 
          (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(home_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_visitor_desc, "S.FOUL") ~ TRUE,
        str_detect(visitor_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_home_desc, "S.FOUL") ~ TRUE,
        (str_detect(home_description, "1 of 1") | str_detect(home_description, "2 of 2") | str_detect(home_description, "3 of 3")) & str_detect(home_description, "PTS") & 
          (!str_detect(Prev_visitor_desc2, "Transition") & !str_detect(Prev_visitor_desc, "Transition") & !str_detect(home_description, "Flagrant")) ~  TRUE,
        (str_detect(visitor_description, "1 of 1") | str_detect(visitor_description, "2 of 2") | str_detect(visitor_description, "3 of 3")) & str_detect(visitor_description, "PTS") & 
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
        str_detect(home_description, "Turnover")  ~ "Home",
        str_detect(visitor_description, "Turnover")   ~ "Away",
        str_detect(home_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_visitor_desc, "S.FOUL")  ~ "Home",
        str_detect(visitor_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_home_desc, "S.FOUL")  ~ "Away",
        (str_detect(home_description, "1 of 1") | str_detect(home_description, "2 of 2") | str_detect(home_description, "3 of 3")) & 
          str_detect(home_description, "PTS") & (!str_detect(Prev_visitor_desc2, "Transition") | 
                                                   !str_detect(Prev_visitor_desc, "Transition") | !str_detect(home_description, "Flagrant"))  ~ "Home",
        (str_detect(visitor_description, "1 of 1") | str_detect(visitor_description, "2 of 2") | str_detect(visitor_description, "3 of 3")) & 
          str_detect(visitor_description, "PTS") & (!str_detect(Prev_home_desc2, "Transition") | 
                                                      !str_detect(Prev_home_desc, "Transition") | !str_detect(visitor_description, "Flagrant"))  ~ "Away",
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
  
  data_stats<-as.data.frame(hoopR::nba_leaguedashplayerstats(league_id= '00',season=Season))
  
  df_selected<-data_stats%>% select(PlayerID= 1,FTPerc= 20,ThreePerc= 17,FTA= 19,ThreePA= 16) %>% mutate_all(as.numeric)
  
  
  NBA24 <- NBA24 %>%
    left_join(df_selected, by = c("player1_id" = "PlayerID")) %>%
    mutate(
      Exp3PT = ThreePerc,  # Create the new column Exp3PT based on selected_df's ThreePerc
      ExpFT = FTPerc
    ) %>%
    select(-ThreePerc, -FTPerc)  # Optionally remove the joined columns if they are no longer needed
  
  
  
  
  NBA24 <- NBA24  %>%
    mutate(
      OffNet = case_when(
        grepl("Free Throw", home_description, ignore.case = TRUE) | grepl("Free Throw", visitor_description, ignore.case = TRUE) | grepl("FreeThrow", event_type, ignore.case = TRUE) ~ (ExpFT),
        (grepl("PTS", home_description, ignore.case = TRUE) & !grepl("3PT", home_description, ignore.case = TRUE) & event_type == "MAKE") | 
          (grepl("PTS", visitor_description, ignore.case = TRUE) & !grepl("3PT", visitor_description, ignore.case = TRUE) & event_type == "MAKE")~ 2,
        grepl("3PT", home_description, ignore.case = TRUE) | grepl("3PT", visitor_description, ignore.case = TRUE) ~ (Exp3PT*3)*0.2+0.8*(Home_Action_Score + Away_Action_Score),
        TRUE ~ 0
      ),
      DefNet = case_when(
        grepl("Free Throw", home_description, ignore.case = TRUE) | grepl("Free Throw", visitor_description, ignore.case = TRUE) | grepl("FreeThrow", event_type, ignore.case = TRUE) ~ (ExpFT),
        (grepl("PTS", home_description, ignore.case = TRUE) & !grepl("3PT", home_description, ignore.case = TRUE) & event_type == "MAKE") | 
          (grepl("PTS", visitor_description, ignore.case = TRUE) & !grepl("3PT", visitor_description, ignore.case = TRUE) & event_type == "MAKE")~ 2,
        grepl("3PT", home_description, ignore.case = TRUE) | grepl("3PT", visitor_description, ignore.case = TRUE) ~ (Exp3PT*3)*0.4+0.6*(Home_Action_Score + Away_Action_Score),
        TRUE ~ 0
      ),
      OffTotal = cumsum(OffNet),
      DefTotal = cumsum(DefNet),
    )
  
  
  View(NBA24)
  
  Player24 <- hoopR::nba_commonallplayers(league_id = '00', season = 2024)
  
  Players = as.data.frame(Player24)
  
  print("Hello")
  Dict24 <- Players[, c(1, 3)]
  
  
  NBA24Filt <- NBA24[NBA24$End_of_Possession == TRUE, ]
  
  NBA24Filt <- NBA24Filt %>%
    mutate(Net_Diff = Net_Total - lag(Net_Total, default = 0))
  
  NBA24Filt <- NBA24Filt %>%
    mutate(Off_Diff = OffTotal - lag(OffTotal, default = 0))
  
  NBA24Filt <- NBA24Filt %>%
    mutate(Def_Diff = DefTotal - lag(DefTotal, default = 0))
  
  for (i in 1:nrow(Dict24)) {
    value_to_replace <- Dict24[i,1]
    replacement_value <- Dict24[i,2]
    
    NBA24Filt[, 78:87] <- lapply(NBA24Filt[, 78:87], function(x) {
      replace(x, x == value_to_replace, replacement_value)
    })
  }
  
  NBA24Filt=NBA24Filt  %>% select(1,78:87, 96:98)
  
  print("Hello")
  
  Schedule = hoopR::nba_schedule(league_id = '00', season = year)
  
  NBA24Filt <- NBA24Filt %>%
    left_join(Schedule %>% mutate(game_id = as.integer(game_id)) %>% select(game_id, game_date), by = "game_id")
  
  NBA24Filt = NBA24Filt[NBA24Filt$D5 != 0,]
  
  NBA24Filt$Season = year+1
  
  return(NBA24Filt)
  
}








ProcessTOs <- function(file_name, year) { 
  NBA24 = read.csv(file_name)
  NBA24 = NBA24 %>% filter(if_all(c("away_player1", "away_player2", "away_player3", "away_player4", "away_player5", "home_player1", "home_player2", "home_player3", "home_player4", "home_player5"), ~ !is.na(.) & . != ""))
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
        str_detect(visitor_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") & str_detect(Prev_visitor_desc, "Free Throw") 
        &  (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
        str_detect(visitor_description, "rebound") & str_detect(Prev_visitor_desc, "MISS") & str_detect(Prev_visitor_desc, "Free Throw") 
        &  (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
        str_detect(home_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") & str_detect(Prev_home_desc, "Free Throw") 
        &  (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
        str_detect(home_description, "rebound") & str_detect(Prev_home_desc, "MISS") & str_detect(Prev_home_desc, "Free Throw") 
        &  (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)"))~ TRUE,
        TRUE ~ FALSE
      ),
      End_of_Possession = case_when(
        str_detect(event_type, "EndOfPeriod") & prev_seconds > 0 ~ TRUE,
        str_detect(home_description, "Flagrant 2 of 2") | str_detect(home_description, "Flagrant 3 of 3") ~ TRUE,
        str_detect(visitor_description, "Flagrant 2 of 2") | str_detect(visitor_description, "Flagrant 3 of 3") ~ TRUE,
        str_detect(event_type, "Turnover") ~ TRUE,
        str_detect(home_description, "REBOUND") & str_detect(Prev_visitor_desc, "MISS") & 
          (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(home_description, "Rebound") & str_detect(Prev_visitor_desc, "MISS") & 
          (!str_detect(Prev_visitor_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(visitor_description, "REBOUND") & str_detect(Prev_home_desc, "MISS") & 
          (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(visitor_description, "Rebound") & str_detect(Prev_home_desc, "MISS") & 
          (!str_detect(Prev_home_desc, "(1 of 2|1 of 3|2 of 3|Technical|Flagrant)")) ~ TRUE,
        str_detect(home_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_visitor_desc, "S.FOUL") ~ TRUE,
        str_detect(visitor_description, "PTS") & str_detect(event_type, "MAKE") & !str_detect(Next_home_desc, "S.FOUL") ~ TRUE,
        (str_detect(home_description, "1 of 1") | str_detect(home_description, "2 of 2") | str_detect(home_description, "3 of 3")) & str_detect(home_description, "PTS") & 
          (!str_detect(Prev_visitor_desc2, "Transition") & !str_detect(Prev_visitor_desc, "Transition") & !str_detect(home_description, "Flagrant")) ~  TRUE,
        (str_detect(visitor_description, "1 of 1") | str_detect(visitor_description, "2 of 2") | str_detect(visitor_description, "3 of 3")) & str_detect(visitor_description, "PTS") & 
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
  
  NBA24 <- propagate_player_values(NBA24,1)
  NBA24 <- FTOffCheck(NBA24,1)
  
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








ProcessRebounding <- function(file_name, year) {
  NBA24 = read.csv(file_name) 
  NBA24 = NBA24 %>% filter(if_all(c("away_player1", "away_player2", "away_player3", "away_player4", "away_player5", "home_player1", "home_player2", "home_player3", "home_player4", "home_player5"), ~ !is.na(.) & . != ""))
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




ProcessTS <- function(file_name, year) {
  NBA24 = read.csv(file_name)
  NBA24 = NBA24 %>% filter(if_all(c("away_player1", "away_player2", "away_player3", "away_player4", "away_player5", "home_player1", "home_player2", "home_player3", "home_player4", "home_player5"), ~ !is.na(.) & . != ""))
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






















































modify_dataframe <- function(df) {
  df <- df[-1]           # Remove the first column of the dataframe
  df$possessions <- 1    # Set all values in the possessions column to 1
  return(df)
}
















RAPM24 = ProcessRAPM("nba24True.csv", 2023, "2023-24")
write.csv(RAPM24, "PlayoffRAPM24.csv", row.names = FALSE)
TO24 = ProcessTOs("nba24True.csv", 2023)
write.csv(TO24, "PlayoffTO24.csv", row.names = FALSE)
Rebounds24 = ProcessRebounding("nba24True.csv", 2023)
write.csv(Rebounds24, "PlayoffRebounds24.csv", row.names = FALSE)
TS24 = ProcessTS("nba24True.csv", 2023)
write.csv(TS24, "PlayoffTS24.csv", row.names = FALSE)

RAPM23 = ProcessRAPM("nba23True.csv", 2022, "2022-23")
write.csv(RAPM23, "PlayoffRAPM23.csv", row.names = FALSE)
TO23 = ProcessTOs("nba23True.csv", 2022)
write.csv(TO23, "PlayoffTO23.csv", row.names = FALSE)
Rebounds23 = ProcessRebounding("nba23True.csv", 2022)
write.csv(Rebounds23, "PlayoffRebounds23.csv", row.names = FALSE)
TS23 = ProcessTS("nba23True.csv", 2022)
write.csv(TS23, "PlayoffTS23.csv", row.names = FALSE)

RAPM22 = ProcessRAPM("nba22True.csv", 2021, "2021-22")
write.csv(RAPM22, "PlayoffRAPM22.csv", row.names = FALSE)
TO22 = ProcessTOs("nba22True.csv", 2021)
write.csv(TO22, "PlayoffTO22.csv", row.names = FALSE)
Rebounds22 = ProcessRebounding("nba22True.csv", 2021)
write.csv(Rebounds22, "PlayoffRebounds22.csv", row.names = FALSE)
TS22 = ProcessTS("nba22True.csv", 2021)
write.csv(TS22, "PlayoffTS22.csv", row.names = FALSE)

RAPM21 = ProcessRAPM("nba21True.csv", 2020, "2020-21")
write.csv(RAPM21, "PlayoffRAPM21.csv", row.names = FALSE)
TO21 = ProcessTOs("nba21True.csv", 2020)
write.csv(TO21, "PlayoffTO21.csv", row.names = FALSE)
Rebounds21 = ProcessRebounding("nba21True.csv", 2020)
write.csv(Rebounds21, "PlayoffRebounds21.csv", row.names = FALSE)
TS21 = ProcessTS("nba21True.csv", 2020)
write.csv(TS21, "PlayoffTS21.csv", row.names = FALSE)

RAPM20 = ProcessRAPM("nba20True.csv", 2019, "2019-20")
write.csv(RAPM20, "PlayoffRAPM20.csv", row.names = FALSE)
TO20 = ProcessTOs("nba20True.csv", 2019)
write.csv(TO20, "PlayoffTO20.csv", row.names = FALSE)
Rebounds20 = ProcessRebounding("nba20True.csv", 2019)
write.csv(Rebounds20, "PlayoffRebounds20.csv", row.names = FALSE)
TS20 = ProcessTS("nba20True.csv", 2019)
write.csv(TS20, "PlayoffTS20.csv", row.names = FALSE)

RAPM19 = ProcessRAPM("nba19True.csv", 2018, "2018-19")
write.csv(RAPM19, "PlayoffRAPM19.csv", row.names = FALSE)
TO19 = ProcessTOs("nba19True.csv", 2018)
write.csv(TO19, "PlayoffTO19.csv", row.names = FALSE)
Rebounds19 = ProcessRebounding("nba19True.csv", 2018)
write.csv(Rebounds19, "PlayoffRebounds19.csv", row.names = FALSE)
TS19 = ProcessTS("nba19True.csv", 2018)
write.csv(TS19, "PlayoffTS19.csv", row.names = FALSE)

RAPM18 = ProcessRAPM("nba18True.csv", 2017, "2017-18")
write.csv(RAPM18, "PlayoffRAPM18.csv", row.names = FALSE)
TO18 = ProcessTOs("nba18True.csv", 2017)
write.csv(TO18, "PlayoffTO18.csv", row.names = FALSE)
Rebounds18 = ProcessRebounding("nba18True.csv", 2017)
write.csv(Rebounds18, "PlayoffRebounds18.csv", row.names = FALSE)
TS18 = ProcessTS("nba18True.csv", 2017)
write.csv(TS18, "PlayoffTS18.csv", row.names = FALSE)

RAPM17 = ProcessRAPM("nba17True.csv", 2016, "2016-17")
write.csv(RAPM17, "PlayoffRAPM17.csv", row.names = FALSE)
TO17 = ProcessTOs("nba17True.csv", 2016)
write.csv(TO17, "PlayoffTO17.csv", row.names = FALSE)
Rebounds17 = ProcessRebounding("nba17True.csv", 2016)
write.csv(Rebounds17, "PlayoffRebounds17.csv", row.names = FALSE)
TS17 = ProcessTS("nba17True.csv", 2016)
write.csv(TS17, "PlayoffTS17.csv", row.names = FALSE)

RAPM16 = ProcessRAPM("nba16True.csv", 2015, "2015-16")
write.csv(RAPM16, "PlayoffRAPM16.csv", row.names = FALSE)
TO16 = ProcessTOs("nba16True.csv", 2015)
write.csv(TO16, "PlayoffTO16.csv", row.names = FALSE)
Rebounds16 = ProcessRebounding("nba16True.csv", 2015)
write.csv(Rebounds16, "PlayoffRebounds16.csv", row.names = FALSE)
TS16 = ProcessTS("nba16True.csv", 2015)
write.csv(TS16, "PlayoffTS16.csv", row.names = FALSE)

RAPM15 = ProcessRAPM("nba15True.csv", 2014, "2014-15")
write.csv(RAPM15, "PlayoffRAPM15.csv", row.names = FALSE)
TO15 = ProcessTOs("nba15True.csv", 2014)
write.csv(TO15, "PlayoffTO15.csv", row.names = FALSE)
Rebounds15 = ProcessRebounding("nba15True.csv", 2014)
write.csv(Rebounds15, "PlayoffRebounds15.csv", row.names = FALSE)
TS15 = ProcessTS("nba15True.csv", 2014)
write.csv(TS15, "PlayoffTS15.csv", row.names = FALSE)

RAPM14 = ProcessRAPM("nba14True.csv", 2013, "2013-14")
write.csv(RAPM14, "PlayoffRAPM14.csv", row.names = FALSE)
TO14 = ProcessTOs("nba14True.csv", 2013)
write.csv(TO14, "PlayoffTO14.csv", row.names = FALSE)
Rebounds14 = ProcessRebounding("nba14True.csv", 2013)
write.csv(Rebounds14, "PlayoffRebounds14.csv", row.names = FALSE)
TS14 = ProcessTS("nba14True.csv", 2013)
write.csv(TS14, "PlayoffTS14.csv", row.names = FALSE)

RAPM13 = ProcessRAPM("nba13True.csv", 2012, "2012-13")
write.csv(RAPM13, "PlayoffRAPM13.csv", row.names = FALSE)
TO13 = ProcessTOs("nba13True.csv", 2012)
write.csv(TO13, "PlayoffTO13.csv", row.names = FALSE)
Rebounds13 = ProcessRebounding("nba13True.csv", 2012)
write.csv(Rebounds13, "PlayoffRebounds13.csv", row.names = FALSE)
TS13 = ProcessTS("nba13True.csv", 2012)
write.csv(TS13, "PlayoffTS13.csv", row.names = FALSE)

RAPM12 = ProcessRAPM("nba12True.csv", 2011, "2011-12")
write.csv(RAPM12, "PlayoffRAPM12.csv", row.names = FALSE)
TO12 = ProcessTOs("nba12True.csv", 2011)
write.csv(TO12, "PlayoffTO12.csv", row.names = FALSE)
Rebounds12 = ProcessRebounding("nba12True.csv", 2011)
write.csv(Rebounds12, "PlayoffRebounds12.csv", row.names = FALSE)
TS12 = ProcessTS("nba12True.csv", 2011)
write.csv(TS12, "PlayoffTS12.csv", row.names = FALSE)

RAPM11 = ProcessRAPM("nba11True.csv", 2010, "2010-11")
write.csv(RAPM11, "PlayoffRAPM11.csv", row.names = FALSE)
TO11 = ProcessTOs("nba11True.csv", 2010)
write.csv(TO11, "PlayoffTO11.csv", row.names = FALSE)
Rebounds11 = ProcessRebounding("nba11True.csv", 2010)
write.csv(Rebounds11, "PlayoffRebounds11.csv", row.names = FALSE)
TS11 = ProcessTS("nba11True.csv", 2010)
write.csv(TS11, "PlayoffTS11.csv", row.names = FALSE)

RAPM10 = ProcessRAPM("nba10True.csv", 2009, "2009-10")
write.csv(RAPM10, "PlayoffRAPM10.csv", row.names = FALSE)
TO10 = ProcessTOs("nba10True.csv", 2009)
write.csv(TO10, "PlayoffTO10.csv", row.names = FALSE)
Rebounds10 = ProcessRebounding("nba10True.csv", 2009)
write.csv(Rebounds10, "PlayoffRebounds10.csv", row.names = FALSE)
TS10 = ProcessTS("nba10True.csv", 2009)
write.csv(TS10, "PlayoffTS10.csv", row.names = FALSE)

RAPM09 = ProcessRAPM("nba09True.csv", 2008, "2008-09")
write.csv(RAPM09, "PlayoffRAPM09.csv", row.names = FALSE)
TO09 = ProcessTOs("nba09True.csv", 2008)
write.csv(TO09, "PlayoffTO09.csv", row.names = FALSE)
Rebounds09 = ProcessRebounding("nba09True.csv", 2008)
write.csv(Rebounds09, "PlayoffRebounds09.csv", row.names = FALSE)
TS09 = ProcessTS("nba09True.csv", 2008)
write.csv(TS09, "PlayoffTS09.csv", row.names = FALSE)

RAPM08 = ProcessRAPM("nba08True.csv", 2007, "2007-08")
write.csv(RAPM08, "PlayoffRAPM08.csv", row.names = FALSE)
TO08 = ProcessTOs("nba08True.csv", 2007)
write.csv(TO08, "PlayoffTO08.csv", row.names = FALSE)
Rebounds08 = ProcessRebounding("nba08True.csv", 2007)
write.csv(Rebounds08, "PlayoffRebounds08.csv", row.names = FALSE)
TS08 = ProcessTS("nba08True.csv", 2007)
write.csv(TS08, "PlayoffTS08.csv", row.names = FALSE)

RAPM07 = ProcessRAPM("nba07True.csv", 2006, "2006-07")
write.csv(RAPM07, "PlayoffRAPM07.csv", row.names = FALSE)
TO07 = ProcessTOs("nba07True.csv", 2006)
write.csv(TO07, "PlayoffTO07.csv", row.names = FALSE)
Rebounds07 = ProcessRebounding("nba07True.csv", 2006)
write.csv(Rebounds07, "PlayoffRebounds07.csv", row.names = FALSE)
TS07 = ProcessTS("nba07True.csv", 2006)
write.csv(TS07, "PlayoffTS07.csv", row.names = FALSE)

RAPM06 = ProcessRAPM("nba06True.csv", 2005, "2005-06")
write.csv(RAPM06, "PlayoffRAPM06.csv", row.names = FALSE)
TO06 = ProcessTOs("nba06True.csv", 2005)
write.csv(TO06, "PlayoffTO06.csv", row.names = FALSE)
Rebounds06 = ProcessRebounding("nba06True.csv", 2005)
write.csv(Rebounds06, "PlayoffRebounds06.csv", row.names = FALSE)
TS06 = ProcessTS("nba06True.csv", 2005)
write.csv(TS06, "PlayoffTS06.csv", row.names = FALSE)

RAPM05 = ProcessRAPM("nba05True.csv", 2004, "2004-05")
write.csv(RAPM05, "PlayoffRAPM05.csv", row.names = FALSE)
TO05 = ProcessTOs("nba05True.csv", 2004)
write.csv(TO05, "PlayoffTO05.csv", row.names = FALSE)
Rebounds05 = ProcessRebounding("nba05True.csv", 2004)
write.csv(Rebounds05, "PlayoffRebounds05.csv", row.names = FALSE)
TS05 = ProcessTS("nba05True.csv", 2004)
write.csv(TS05, "PlayoffTS05.csv", row.names = FALSE)

RAPM04 = ProcessRAPM("nba04True.csv", 2003, "2003-04")
write.csv(RAPM04, "PlayoffRAPM04.csv", row.names = FALSE)
TO04 = ProcessTOs("nba04True.csv", 2003)
write.csv(TO04, "PlayoffTO04.csv", row.names = FALSE)
Rebounds04 = ProcessRebounding("nba04True.csv", 2003)
write.csv(Rebounds04, "PlayoffRebounds04.csv", row.names = FALSE)
TS04 = ProcessTS("nba04True.csv", 2003)
write.csv(TS04, "PlayoffTS04.csv", row.names = FALSE)

RAPM03 = ProcessRAPM("nba03True.csv", 2002, "2002-03")
write.csv(RAPM03, "PlayoffRAPM03.csv", row.names = FALSE)
TO03 = ProcessTOs("nba03True.csv", 2002)
write.csv(TO03, "PlayoffTO03.csv", row.names = FALSE)
Rebounds03 = ProcessRebounding("nba03True.csv", 2002)
write.csv(Rebounds03, "PlayoffRebounds03.csv", row.names = FALSE)
TS03 = ProcessTS("nba03True.csv", 2002)
write.csv(TS03, "PlayoffTS03.csv", row.names = FALSE)

RAPM02 = ProcessRAPM("nba02True.csv", 2001, "2001-02")
write.csv(RAPM02, "PlayoffRAPM02.csv", row.names = FALSE)
TO02 = ProcessTOs("nba02True.csv", 2001)
write.csv(TO02, "PlayoffTO02.csv", row.names = FALSE)
Rebounds02 = ProcessRebounding("nba02True.csv", 2001)
write.csv(Rebounds02, "PlayoffRebounds02.csv", row.names = FALSE)
TS02 = ProcessTS("nba02True.csv", 2001)
write.csv(TS02, "PlayoffTS02.csv", row.names = FALSE)

RAPM01 = ProcessRAPM("nba01True.csv", 2000, "2000-01")
write.csv(RAPM01, "PlayoffRAPM01.csv", row.names = FALSE)
TO01 = ProcessTOs("nba01True.csv", 2000)
write.csv(TO01, "PlayoffTO01.csv", row.names = FALSE)
Rebounds01 = ProcessRebounding("nba01True.csv", 2000)
write.csv(Rebounds01, "PlayoffRebounds01.csv", row.names = FALSE)
TS01 = ProcessTS("nba01True.csv", 2000)
write.csv(TS01, "PlayoffTS01.csv", row.names = FALSE)

RAPM00 = ProcessRAPM("nba00True.csv", 1999, "1999-00")
write.csv(RAPM00, "PlayoffRAPM00.csv", row.names = FALSE)
TO00 = ProcessTOs("nba00True.csv", 1999)
write.csv(TO00, "PlayoffTO00.csv", row.names = FALSE)
Rebounds00 = ProcessRebounding("nba00True.csv", 1999)
write.csv(Rebounds00, "PlayoffRebounds00.csv", row.names = FALSE)
TS00 = ProcessTS("nba00True.csv", 1999)
write.csv(TS00, "PlayoffTS00.csv", row.names = FALSE)

# Manually reading in all the CSVs for turnovers, rebounds, TS, and RAPM for each year

# For Turnovers
TO00 = read.csv("PlayoffTO00.csv")
TO01 = read.csv("PlayoffTO01.csv")
TO02 = read.csv("PlayoffTO02.csv")
TO03 = read.csv("PlayoffTO03.csv")
TO04 = read.csv("PlayoffTO04.csv")
TO05 = read.csv("PlayoffTO05.csv")
TO06 = read.csv("PlayoffTO06.csv")
TO07 = read.csv("PlayoffTO07.csv")
TO08 = read.csv("PlayoffTO08.csv")
TO09 = read.csv("PlayoffTO09.csv")
TO10 = read.csv("PlayoffTO10.csv")
TO11 = read.csv("PlayoffTO11.csv")
TO12 = read.csv("PlayoffTO12.csv")
TO13 = read.csv("PlayoffTO13.csv")
TO14 = read.csv("PlayoffTO14.csv")
TO15 = read.csv("PlayoffTO15.csv")
TO16 = read.csv("PlayoffTO16.csv")
TO17 = read.csv("PlayoffTO17.csv")
TO18 = read.csv("PlayoffTO18.csv")
TO19 = read.csv("PlayoffTO19.csv")
TO20 = read.csv("PlayoffTO20.csv")
TO21 = read.csv("PlayoffTO21.csv")
TO22 = read.csv("PlayoffTO22.csv")
TO23 = read.csv("PlayoffTO23.csv")
TO24 = read.csv("PlayoffTO24.csv")

# For Rebounds
Rebounds00 = read.csv("PlayoffRebounds00.csv")
Rebounds01 = read.csv("PlayoffRebounds01.csv")
Rebounds02 = read.csv("PlayoffRebounds02.csv")
Rebounds03 = read.csv("PlayoffRebounds03.csv")
Rebounds04 = read.csv("PlayoffRebounds04.csv")
Rebounds05 = read.csv("PlayoffRebounds05.csv")
Rebounds06 = read.csv("PlayoffRebounds06.csv")
Rebounds07 = read.csv("PlayoffRebounds07.csv")
Rebounds08 = read.csv("PlayoffRebounds08.csv")
Rebounds09 = read.csv("PlayoffRebounds09.csv")
Rebounds10 = read.csv("PlayoffRebounds10.csv")
Rebounds11 = read.csv("PlayoffRebounds11.csv")
Rebounds12 = read.csv("PlayoffRebounds12.csv")
Rebounds13 = read.csv("PlayoffRebounds13.csv")
Rebounds14 = read.csv("PlayoffRebounds14.csv")
Rebounds15 = read.csv("PlayoffRebounds15.csv")
Rebounds16 = read.csv("PlayoffRebounds16.csv")
Rebounds17 = read.csv("PlayoffRebounds17.csv")
Rebounds18 = read.csv("PlayoffRebounds18.csv")
Rebounds19 = read.csv("PlayoffRebounds19.csv")
Rebounds20 = read.csv("PlayoffRebounds20.csv")
Rebounds21 = read.csv("PlayoffRebounds21.csv")
Rebounds22 = read.csv("PlayoffRebounds22.csv")
Rebounds23 = read.csv("PlayoffRebounds23.csv")
Rebounds24 = read.csv("PlayoffRebounds24.csv")

# For True Shooting (TS)
TS00 = read.csv("PlayoffTS00.csv")
TS01 = read.csv("PlayoffTS01.csv")
TS02 = read.csv("PlayoffTS02.csv")
TS03 = read.csv("PlayoffTS03.csv")
TS04 = read.csv("PlayoffTS04.csv")
TS05 = read.csv("PlayoffTS05.csv")
TS06 = read.csv("PlayoffTS06.csv")
TS07 = read.csv("PlayoffTS07.csv")
TS08 = read.csv("PlayoffTS08.csv")
TS09 = read.csv("PlayoffTS09.csv")
TS10 = read.csv("PlayoffTS10.csv")
TS11 = read.csv("PlayoffTS11.csv")
TS12 = read.csv("PlayoffTS12.csv")
TS13 = read.csv("PlayoffTS13.csv")
TS14 = read.csv("PlayoffTS14.csv")
TS15 = read.csv("PlayoffTS15.csv")
TS16 = read.csv("PlayoffTS16.csv")
TS17 = read.csv("PlayoffTS17.csv")
TS18 = read.csv("PlayoffTS18.csv")
TS19 = read.csv("PlayoffTS19.csv")
TS20 = read.csv("PlayoffTS20.csv")
TS21 = read.csv("PlayoffTS21.csv")
TS22 = read.csv("PlayoffTS22.csv")
TS23 = read.csv("PlayoffTS23.csv")
TS24 = read.csv("PlayoffTS24.csv")

# For RAPM
RAPM00 = read.csv("PlayoffRAPM00.csv")
RAPM01 = read.csv("PlayoffRAPM01.csv")
RAPM02 = read.csv("PlayoffRAPM02.csv")
RAPM03 = read.csv("PlayoffRAPM03.csv")
RAPM04 = read.csv("PlayoffRAPM04.csv")
RAPM05 = read.csv("PlayoffRAPM05.csv")
RAPM06 = read.csv("PlayoffRAPM06.csv")
RAPM07 = read.csv("PlayoffRAPM07.csv")
RAPM08 = read.csv("PlayoffRAPM08.csv")
RAPM09 = read.csv("PlayoffRAPM09.csv")
RAPM10 = read.csv("PlayoffRAPM10.csv")
RAPM11 = read.csv("PlayoffRAPM11.csv")
RAPM12 = read.csv("PlayoffRAPM12.csv")
RAPM13 = read.csv("PlayoffRAPM13.csv")
RAPM14 = read.csv("PlayoffRAPM14.csv")
RAPM15 = read.csv("PlayoffRAPM15.csv")
RAPM16 = read.csv("PlayoffRAPM16.csv")
RAPM17 = read.csv("PlayoffRAPM17.csv")
RAPM18 = read.csv("PlayoffRAPM18.csv")
RAPM19 = read.csv("PlayoffRAPM19.csv")
RAPM20 = read.csv("PlayoffRAPM20.csv")
RAPM21 = read.csv("PlayoffRAPM21.csv")
RAPM22 = read.csv("PlayoffRAPM22.csv")
RAPM23 = read.csv("PlayoffRAPM23.csv")
RAPM24 = read.csv("PlayoffRAPM24.csv")



turnovers_list = list(TO00,TO01,TO02,TO03,TO04,TO05,TO06,TO07,TO08,TO09,TO10,TO11,
                      TO12,TO13,TO14,TO15,TO16,TO17,TO18,TO19,TO20,TO21,TO22,TO23,TO24)

rebounds_list = list(Rebounds00,Rebounds01,Rebounds02,Rebounds03,Rebounds04,Rebounds05,Rebounds06,Rebounds07,Rebounds08,Rebounds09,Rebounds10,Rebounds11,Rebounds12,
                     Rebounds13,Rebounds14,Rebounds15,Rebounds16,Rebounds17,Rebounds18,Rebounds19,Rebounds20,Rebounds21,Rebounds22,Rebounds23,Rebounds24)

ts_list = list(TS00,TS01,TS02,TS03,TS04,TS05,TS06,TS07,TS08,TS09,TS10,TS11,
               TS12,TS13,TS14,TS15,TS16,TS17,TS18,TS19,TS20,TS21,TS22,TS23,TS24)

rapm_list = list(RAPM00,RAPM01,RAPM02,RAPM03,RAPM04,RAPM05,RAPM06,RAPM07,RAPM08,RAPM09,RAPM10,RAPM11,
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
Combined_rapm <- do.call(rbind, modified_rapm)




library(RMariaDB)

#Connect to Database
con <- dbConnect(RMariaDB::MariaDB(),
                 host = "127.0.0.1",             # Host (localhost in this case)
                 dbname = "my_new_database",     # Database name
                 user = "root",                  # Username
                 password = "tijaytijay",        # Password
                 port = 3306)                    # Default MySQL port

dbWriteTable(con, "PlayoffTOFULLTABLE", Combined_Turnovers, overwrite = TRUE, row.names = FALSE)
dbDisconnect(con)

# Write the dataframe 'NBAPOSSTOTAL' to the SQL database as a new table named 'NBASTATS'
dbWriteTable(con, "PlayoffTSFULLTABLE", Combined_Turnovers, overwrite = TRUE, row.names = FALSE)
dbWriteTable(con, "PlayoffREBFULLTABLE",Combined_Rebounds, overwrite = TRUE, row.names = FALSE)
dbWriteTable(con, "PlayoffTSFULLTABLE", Combined_TS, overwrite = TRUE, row.names = FALSE)
dbWriteTable(con, "PlayoffRAPMFULLTABLE", Combined_rapm, overwrite = TRUE, row.names = FALSE)



# Close the connection after the operation
dbDisconnect(con)
















