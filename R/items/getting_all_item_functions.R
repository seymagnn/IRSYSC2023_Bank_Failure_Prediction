# Getting all banks with getInstitutionAll()
# bank_infos <- getInstitutionsAll() 
bank_infos <- read_csv("institutions.csv")
bank_infos <- bank_infos %>% filter(ACTIVE == 1)

failure_banklist <- read.csv("banklist.csv", header=FALSE, row.names=NULL)
names(failure_banklist) <- c("Bank Name","City","State","Cert","Acquiring Institution","Closing Date","Fund")
failure_banklist <- failure_banklist[-1,] 

getAllFinancials <- function(IDRSSD_or_CERT, metrics, limit=1,IDRSSD = TRUE) {
  if(IDRSSD == TRUE){
    url <- paste0(
      "https://banks.data.fdic.gov/api/financials?filters=RSSDID%3A%20",
      IDRSSD_or_CERT,
      "&fields=RSSDID%2CREPDTE%2C",
      paste(metrics, collapse = '%2C'),
      "&sort_by=REPDTE&sort_order=DESC&limit=",
      limit,
      "&offset=0&agg_term_fields=REPDTE&agg_limit=1&format=csv&download=false&filename=data_file"
    )
  }
  if(IDRSSD == FALSE){
    url <- paste0(
      "https://banks.data.fdic.gov/api/financials?filters=CERT%3A%20",
      IDRSSD_or_CERT,
      "&fields=RSSDID%2CREPDTE%2C",
      paste(c(metrics,"RSSDID","CERT"), collapse = '%2C'),
      "&sort_by=REPDTE&sort_order=DESC&limit=",
      limit,
      "&offset=0&agg_term_fields=REPDTE&agg_limit=1&format=csv&download=false&filename=data_file"
    )
  }
  
  df <- read.csv(url,header=TRUE)
  
  df <- df %>%
    mutate(
      ID = NULL,
      Date =  as.Date(as.character(REPDTE), "%Y%m%d")
    ) %>%
    select(-REPDTE) %>%
    rename("IDRSSD" = "RSSDID")
  return(df)
}



getAllFinancialsLoop <- function(IDRSSD_or_CERT,metrics,getAllFinancials,IDRSSD = TRUE,limit=100){
  
  all_financials_banks <- data.frame()
  feds <- IDRSSD_or_CERT
  n_feds <- length(feds)
  start_time <- Sys.time()
  for (j in 1:length(IDRSSD_or_CERT)) {
    i <- feds[j]
    message(paste("Processing", i, "(", j, "of", n_feds, ")"))
    retry <- 0
    success <- FALSE
    while (!success && retry < 3) {
      tryCatch({
        suppressWarnings({
          all_financials_single_bank <<- getAllFinancials(i, metrics, limit = limit,IDRSSD = IDRSSD)
        })
        all_financials_banks <- rbind(all_financials_single_bank, all_financials_banks)
        message(paste(i, "added to data"))
        success <- TRUE
      }, error = function(e) {
        message(paste("API rejected the request, retrying in 3 minutes..."))
        Sys.sleep( sample(1:30, 1))
        retry <<- retry + 1
      })
    }
    if (!success) {
      message(paste("Could not add", i, "to data after 3 tries"))
    }
  }
  end_time <- Sys.time()
  return(all_financials_banks)
  cat(end_time-start_time)
}



