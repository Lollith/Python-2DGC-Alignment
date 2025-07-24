ImportFile <- function(File) { 

  MissingStandards <- c()
  currentRawFile <- read.table(File, sep = "\t", fill = T, 
                               quote = "", strip.white = T, stringsAsFactors = F, 
                               header = T)

  currentRawFile[, 5] <- as.character(currentRawFile[, 
                                                     5])
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



GenerateSimFrames <- function(Sample, SeedSample, RT2Penalty=5, RT1Penalty=1) {
    # ajouter 0 si spectre pas de la meme taille
    mzSeed<-SeedSample[[4]]
    mzSample<-Sample[[4]]
    # cat(mzSeed, mzSample, "\n")
    # print(mzSeed == mzSample)
    

    seedSpectraFrame <- do.call(cbind, SeedSample[[2]])
    seedSpectraFrame <- t(seedSpectraFrame)
    seedSpectraFrame <- as.matrix(seedSpectraFrame)/sqrt(apply((as.matrix(seedSpectraFrame))^2, 
                                                               1, sum))

    # print(paste("Seed spectra shape:", dim(seedSpectraFrame)))

    sampleSpectraFrame <- do.call(cbind, Sample[[2]])
    sampleSpectraFrame <- t(sampleSpectraFrame)
    sampleSpectraFrame <- as.matrix(sampleSpectraFrame)/sqrt(apply((as.matrix(sampleSpectraFrame))^2, 
                                                                   1, sum))

    # print(paste("Sample spectra shape:", dim(sampleSpectraFrame)))

    SimilarityMatrix <- (seedSpectraFrame %*% t(sampleSpectraFrame)) * 100 
        # similarity score entre chaque peak des deux sample s

    RT1Index <- matrix(unlist(lapply(Sample[[1]][, "RT1"], 
                                     function(x) abs(x - SeedSample[[1]][, "RT1"]) * RT1Penalty)), 
                       nrow = nrow(SimilarityMatrix))

    RT2Index <- matrix(unlist(lapply(Sample[[1]][, "RT2"], 
                                     function(x) abs(x - SeedSample[[1]][, "RT2"]) * RT2Penalty)), 
                       nrow = nrow(SimilarityMatrix))

    return(SimilarityMatrix - RT1Index - RT2Index)
}


listFiles <- list("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751303_v3_E3AM_5jui.txt", 
                       "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751304_v1_E3AM_4jui.txt",
                        "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751306_v1_E3PM_5jui.txt")
ImportedFiles <- lapply(listFiles, ImportFile)
#seed est le 1er fichier de la liste
SeedSample <- ImportedFiles[[1]]

print ("seedSample:")
print(SeedSample[[4]])


for (SampNum in (1:length(ImportedFiles))) {
    if (SampNum != 1) { #sauf la seed sample
        SimCutoffs <- GenerateSimFrames(ImportedFiles[[SampNum]], SeedSample)
        # print(SimCutoffs)
        write.table(SimCutoffs, file = paste0("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/", "R_SimCutoffs_", SampNum, ".txt"), sep = "\t", row.names = FALSE, col.names = FALSE)
    }
}



