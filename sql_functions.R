db_connect <- function(server) {
  sql_servers <- data.frame(driver = rep("SQL Server")
                           ,server = c("192.168.126.42\\PremierOne,5453", "DCP-FHSQL1PRD", "DCP-GISSQL1PRD", "DCP-FIRESQL1PRD", "DCP-FDGAPP1DEV", "DCP-FDGAPP1PRD")
                           ,port = c(5453, 1433, 1433, 1433, 1433, 1433)
                           ,database = c("master", "master", "master", "master", "utility", "utility")
                           ,keyService = c(key_list()$service[4]
                                           ,key_list()$service[5]
                                           ,key_list()$service[6]
                                           ,key_list()$service[7]
                                           ,key_list()$service[8]
                                           ,key_list()$service[9])
                           ,key_username = c(key_list()$username[4]
                                            ,key_list()$username[5]
                                            ,key_list()$username[6]
                                            ,key_list()$username[7]
                                            ,key_list()$username[8]
                                            ,key_list()$username[9]))
  
  if (!is.na(match(server, sql_servers$server))) {
    con <- {dbConnect(odbc::odbc(), 
                      Driver   = sql_servers$driver[match(server, sql_servers$server)],
                      Server   = sql_servers$server[match(server, sql_servers$server)],
                      Port     = as.integer(sql_servers$port[match(server, sql_servers$server)]),
                      Database = sql_servers$database[match(server, sql_servers$server)],
                      UID      = keyring::key_list(sql_servers$keyService[match(server, sql_servers$server)])[1,2],
                      PWD      = keyring::key_get(sql_servers$keyService[match(server, sql_servers$server)], sql_servers$key_username[match(server, sql_servers$server)]))}
    print("SQL Connection Established")
    return(con)
    } else {
    stop("Invalid SQL Server Name")
  }
}


db_close <- function(con) {
  dbDisconnect(con)
  print("SQL Connection Closed")
}


get_sql_script_from_file <- function(filepath) {
  con <- file(filepath, "r")
  sql_string <- ""
  
  while (TRUE) {
    line <- readLines(con, n = 1)
    
    if ( length(line) == 0 ){
      break
    }
    
    line <- gsub("\\t", " ", line)
    
    if(grepl("--",line) == TRUE){
      line <- paste(sub("--","/*",line),"*/")
    }
    sql_string <- paste(sql_string, line)
  }
  close(con)
  return(sql_string)
}


execute_sql <- function (con, sql, params = NULL) {
  #params <- c(...)
  if (!is.null(params)) {
    query <- dbSendQuery(con, sql)
    dbBind(query, params)
    return(dbFetch(query))
  } else {
    return(dbGetQuery(con, sql))
  }
}


query_sql <- function(server, sql, ...) {
  params <- c(...)
  con <- db_connect(server)
  if(file.exists(sql) & file_ext(sql) == "sql") {
    sql_script <- get_sql_script_from_file(sql)
  } else {
    sql_script <- sql
  }
  print("Running Query...")
  query_results <- execute_sql(con, sql_script, params)
  db_close(con)
  return(query_results)
}
