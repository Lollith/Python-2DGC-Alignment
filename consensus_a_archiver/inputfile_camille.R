ImportFile <- function(File) { 

#   MissingStandards <- c()
  currentRawFile <- read.table(File, sep = "\t", fill = TRUE, 
                               quote = "", strip.white = TRUE, stringsAsFactors = FALSE, 
                               header = TRUE)

  currentRawFile[, 5] <- as.character(currentRawFile[, 5])
  # currentRawFile<-currentRawFile[which(!is.na(currentRawFile[,3])&nchar(currentRawFile[,5])!=0),]
  currentRawFile[, 2] <- as.character(currentRawFile[, 
2])
  # #  Filtrage des lignes avec valeurs manquantes
  # currentRawFile <- currentRawFile[which(!is.na(currentRawFile[,3]) & 
  #                                       nchar(currentRawFile[,5]) != 0),]

  RTSplit <- data.frame(strsplit(currentRawFile[, 2], " , "), 
                        stringsAsFactors = F)

  RTSplit[1, ] <- gsub("\"", "", RTSplit[1, ])
  RTSplit[2, ] <- gsub("\"", "", RTSplit[2, ])
  currentRawFile[, "RT1"] <- as.numeric(t(RTSplit[1, ]))
  currentRawFile[, "RT2"] <- as.numeric(t(RTSplit[2, ]))

  uniqueIndex <- data.frame(paste(currentRawFile[, 1], 
                                  currentRawFile[, 2], currentRawFile[, 3]))
  currentRawFile <- currentRawFile[which(!duplicated(uniqueIndex)), 
  ]
  row.names(currentRawFile) <- c(1:nrow(currentRawFile))


  currentRawFileSplit <- split(currentRawFile, 1:nrow(currentRawFile))
  spectraSplit <- lapply(currentRawFileSplit, function(a) strsplit(a[[5]], 
                                                                   " "))
  spectraSplit <- lapply(spectraSplit, function(b) lapply(b, 
                                                          function(c) strsplit(c, ":")))

  spectraSplit <- lapply(spectraSplit, function(d) t(matrix(unlist(d), 
                                                            nrow = 2)))

  

  spectraSplit <- lapply(spectraSplit, function(d) apply(d, 
                                                         2, as.numeric))

  ionNames <- spectraSplit[[1]][order(spectraSplit[[1]][, 
                                                        1]), 1]

  spectraSplit <- lapply(spectraSplit, function(d) d[order(d[, 
                                                             1]), 2, drop = F])

  return(list(currentRawFile, spectraSplit, MissingStandards, 
              ionNames, spectraSplit))

}
