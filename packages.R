require_packages <- function(r, py = NULL){
  rInst <- c()
  
  if(length(r) > 0) {
    for(i in r){
      if(!require(i, character.only = TRUE)){
        install.packages(i , dependencies = TRUE)
        require(i , character.only = TRUE)
        rInst <- c(rInst, i)
        
      } else {
        require(i , character.only = TRUE)
      }
    }
    cat("\n### R Packages #######################################################\n")
    if (!is.null(rInst)) cat("\nR Packages Installed: ", paste0(rInst, collapse = ", "), "\n")
    if (!is.null(r)) cat("\nR Packages Loaded: ", paste0(r, collapse = ", "), "\n\n")
  }
  
  i <- NULL
  if(!is.null(py)) {
    cat("\n### Python Packages ##################################################\n")
    cat("\nAttemping Install on Python Packages: ", paste0(py, collapse = ", "), "...\n\n")
    for(i in py){
      py_install(i)
    }
    
  }
}