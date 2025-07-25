#' Takes a vector of paths to input files and aligns common metabolites into a final table.  Will also identify metabolites if a reference library is provided

#' @param inputFileList Vector of file paths to align
#' @param RT1_Standards Vector of standard names used to adjust first retention time. All names must be found in input files. Defaults to NULL.
#' @param RT2_Standards Vector of standard names used to adjust second retention time. All names must be found in input files. Defaults to NULL.
#' @param seedFile File number in inputFileList to initialize alignment. Can also input a vector of different seed files (3 is usually sufficient) to prevent bias from seed file.  Defaults to 1.
#' @param RT1Penalty Penalty used for first retention time errors.  Defaults to 1.
#' @param RT2Penalty Penalty used for first retention time errors.  Defaults to 10.
#' @param autoTuneMatchStringency Will automatically find optimal match threshold. If TRUE, will ignore similarityCutoff. Defaults to TRUE.
#' @param similarityCutoff Adjusts peak similarity threshold required for alignment. Adjust in concordance with RT1 and RT2 penalties. Will be ignored if autoTuneMatchStrigency is TRUE. Defaults to 90.
#' @param disimilarityCutoff Defaults to similarityCutoff-90. Sets the threshold for including a new peak in the alignment table to ensure new metabolites aren't just below alignment thresholds
#' @param numCores Number of cores used to parallelize alignment. See parallel package. Defaults to 4.
#' @param commonIons Provide a vector of ions to ignore from the FindProblemIons function. Defaults to empty vector.
#' @param missingValueLimit Maximum fraction (Numeric between 0 and 1) of missing values acceptable for retaining a metabolite in the final alignment table. Defaults to 0.25.
#' @param missingPeakFinderSimilarityLax Fraction of Similarity Cutoff to use to find missing alignments just below threshold. Set to 1 to prevent searching for missing peaks. Defaults to 0.85.
#' @param quantMethod Set to U, A, or T to indicate if unique mass (U), appexing masses (A), or total ion chormatograph (T) was used to quantify peak areas. Defaults to T.  If "T" or "A", peaks meeting similarity thresholds will simply be summed. If "U", peaks with the same unique mass with be summed and a proportional conversion will be used before combining peaks with different unique masses.
#' @param standardLibrary Defaults to NULL. Provide standard library generated from MakeReference function to ID metabolites with retention index.

#' @return A list with three items: AlignmentMatix - A dataframe with peak areas for all metabolites matched in sufficient number of samples. MetaboliteInfo - An info file with RT, spectra, and metabolite ID info for each metabolite in the AlignmentMatrix. UnmatchedQuantMasses- Info on metabolites combined that had different unique masses (if quantMethod="U") or greater than 50% different apexing masses (if quantMethod="A")
#' @import parallel
#' @import stats
#' @export
#' @examples
#' ConsensusAlign(c(system.file("extdata", "SampleA.txt", package="R2DGC"),
#'     system.file("extdata", "SampleB.txt", package="R2DGC")), RT1_Standards= c())

library(foreach)
library(doParallel)

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

ConsensusAlignBis<-function (inputFileList,
                              ImportedFiles,
                              seedFile=1,
                              RT1_Standards = NULL,
                              RT2_Standards = NULL,
                              c = 1,
                              RT1Penalty = 1,
                              RT2Penalty = 10,
                              autoTuneMatchStringency = TRUE,
                              similarityCutoff = 90,
                              disimilarityCutoff = similarityCutoff - 90,
                              numCores = 1,
                              commonIons = c(),
                              missingValueLimit = 0.75,
                              missingPeakFinderSimilarityLax = 0.85,
                              quantMethod = "T", 
                              standardLibrary = NULL) 
{

  #ne marche pas sur windows
  # ImportedFiles <- mclapply(inputFileList, ImportFile, mc.cores = numCores)

  cl <- parallel::makeCluster(numCores)
  doParallel::registerDoParallel(cl)
  parallel::clusterExport(cl, varlist = c("ImportFile"))

  ImportedFiles <- foreach::foreach(file = inputFileList) %dopar% {ImportFile(file)}
  parallel::stopCluster(cl)


  MissingFileList <- c()
  for (File in ImportedFiles) {
    MissingFileList <- c(MissingFileList, File[3])
  }

  if (length(unlist(MissingFileList)) > 0) {
    stop(unlist(MissingFileList), call. = FALSE)
  }

  GenerateSimFrames <- function(Sample, SeedSample) {
    # ajouter 0 si spectre pas de la meme taille
    mzSeed<-SeedSample[[4]]
    mzSample<-Sample[[4]]
    

    seedSpectraFrame <- do.call(cbind, SeedSample[[2]])
    seedSpectraFrame <- t(seedSpectraFrame)
    seedSpectraFrame <- as.matrix(seedSpectraFrame)/sqrt(apply((as.matrix(seedSpectraFrame))^2, 
                                                               1, sum))

    sampleSpectraFrame <- do.call(cbind, Sample[[2]])
    sampleSpectraFrame <- t(sampleSpectraFrame)
    sampleSpectraFrame <- as.matrix(sampleSpectraFrame)/sqrt(apply((as.matrix(sampleSpectraFrame))^2, 
                                                                   1, sum))

    SimilarityMatrix <- (seedSpectraFrame %*% t(sampleSpectraFrame)) * 
      100 # similarity score entre chaque peak des deux sample s

    RT1Index <- matrix(unlist(lapply(Sample[[1]][, "RT1"], 
                                     function(x) abs(x - SeedSample[[1]][, "RT1"]) * RT1Penalty)), 
                       nrow = nrow(SimilarityMatrix))

    RT2Index <- matrix(unlist(lapply(Sample[[1]][, "RT2"], 
                                     function(x) abs(x - SeedSample[[1]][, "RT2"]) * RT2Penalty)), 
                       nrow = nrow(SimilarityMatrix))

    return(SimilarityMatrix - RT1Index - RT2Index)
  }

  
  #seed est le 1er fichier de la liste
  SeedSample <- ImportedFiles[[seedFile]]

  #initialisation 
  FinalMatrix <- matrix(nrow = nrow(SeedSample[[1]]), ncol = length(inputFileList))
  FinalMatrixRT <- matrix(nrow = nrow(SeedSample[[1]]), ncol = length(inputFileList))
  FinalMatrixSpectra <- matrix(nrow = nrow(SeedSample[[1]]), ncol = length(inputFileList))
  row.names(FinalMatrix) <- paste0(SeedSample[[1]][, 1],"_1")
  colnames(FinalMatrix) <- inputFileList
  row.names(FinalMatrixRT) <- paste0(SeedSample[[1]][, 1],"_1")
  colnames(FinalMatrixRT) <- inputFileList
  row.names(FinalMatrixSpectra) <- paste0(SeedSample[[1]][, 1],"_1")
  colnames(FinalMatrixSpectra) <- inputFileList


  for (SampNum in (1:length(ImportedFiles))) {
    SimCutoffs <- GenerateSimFrames(ImportedFiles[[SampNum]],SeedSample)

    MatchScores <- apply(SimCutoffs, 
                         2, function(x) max(x, na.rm = T))

    Mates <- apply(SimCutoffs, 
                   2, function(x) which.max(x))

    
    dissmatch<- which(MatchScores < disimilarityCutoff)

    
    names(MatchScores) <- 1:length(MatchScores)
    names(Mates) <- 1:length(Mates)
    Mates <- Mates[order(-MatchScores)]
    MatchScores <- MatchScores[order(-MatchScores)]
    MatchScores[which(duplicated(Mates))] <- NA
    Mates <- Mates[order(as.numeric(names(Mates)))]
    MatchScores <- MatchScores[order(as.numeric(names(MatchScores)))]

    if (quantMethod == "T"){
      FinalMatrix[Mates[which(MatchScores >= similarityCutoff)], 
                  inputFileList[SampNum]] <- ImportedFiles[[SampNum]][[1]][which(MatchScores >= similarityCutoff), 3]

      FinalMatrixRT[Mates[which(MatchScores >= similarityCutoff)], 
                    inputFileList[SampNum]] <- ImportedFiles[[SampNum]][[1]][which(MatchScores >= similarityCutoff), 2]

      FinalMatrixSpectra[Mates[which(MatchScores >= similarityCutoff)], 
                         inputFileList[SampNum]] <- ImportedFiles[[SampNum]][[1]][which(MatchScores >= similarityCutoff), 4]

    }

    
    if (length(dissmatch) > 0) {
      SeedSample[[1]] <- rbind(SeedSample[[1]], ImportedFiles[[SampNum]][[1]][dissmatch, ])
      SeedSample[[2]][as.character((length(SeedSample[[2]]) + 1):(length(SeedSample[[2]]) + length(dissmatch)))] <- 
        ImportedFiles[[SampNum]][[2]][dissmatch]

      
      new_rows <- matrix(NA, nrow = length(dissmatch), ncol = ncol(FinalMatrix))
      rownames(new_rows) <- paste0(ImportedFiles[[SampNum]][[1]][dissmatch, "Name"],"_",SampNum)

      
      new_rows_Area<-new_rows_RT<-new_rows_Spectra<-new_rows
      new_rows_Area[,SampNum]<- ImportedFiles[[SampNum]][[1]][dissmatch, "Area"]
      new_rows_RT[,SampNum]<- ImportedFiles[[SampNum]][[1]][dissmatch, "R.T...s."]
      new_rows_Spectra[,SampNum]<- ImportedFiles[[SampNum]][[1]][dissmatch, "Spectra"]
      

      FinalMatrix<-rbind(FinalMatrix,new_rows_Area)
      FinalMatrixRT<-rbind(FinalMatrixRT,new_rows_RT)
      FinalMatrixSpectra<-rbind(FinalMatrix,FinalMatrixSpectra)
    }
  }

  

  SeedSample[[1]]$Name<-rownames(FinalMatrix)

  orderRT<- order(SeedSample[[1]]$RT1)

  

  returnList <- list(FinalMatrix[orderRT,], SeedSample[[1]][orderRT,], FinalMatrixRT[orderRT,],FinalMatrixSpectra[orderRT,])
  names(returnList) <- c("Alignment_Matrix", "Peak_Info", "RT_group","spectra_group")
  return(returnList)
}


# file<-list.files("C:/Users/camil/data/td-ptr/gcxgc/resultPersistantHomology_tic",pattern = ".txt",full.names = TRUE,recursive = TRUE)
# for (sample in file) {
#   PrecompressFiles(inputFileList=sample, outputFiles=T, RT1Penalty=1, RT2Penalty=10,similarityCutoff = 95,quantMethod = "T")
# }


# file<-list.files("C:/Users/camil/data/td-ptr/gcxgc/resultPersistantHomology_tic",pattern = "Processed.txt",full.names = TRUE,recursive = TRUE)
# file <- c("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751303_v3_E3AM_5jui.txt", 
#                        "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751304_v1_E3AM_4jui.txt",
#                         "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/751306_v1_E3PM_5jui.txt")
file <-c("C:\\Users\\adeli\\Documents\\programmation\\uvsq\\Python-2DGC-Alignment\\consensus\\751303_v3_E3AM_5jui.txt", 
         "C:\\Users\\adeli\\Documents\\programmation\\uvsq\\Python-2DGC-Alignment\\consensus\\751304_v1_E3AM_4jui.txt",
         "C:\\Users\\adeli\\Documents\\programmation\\uvsq\\Python-2DGC-Alignment\\consensus\\751306_v1_E3PM_5jui.txt")

Alignment<-ConsensusAlignBis(inputFileList = file, seedFile =1,
  missingValueLimit=0, RT2Penalty = 5, RT1Penalty=1, similarityCutoff=90,
  numCores=6,
  disimilarityCutoff = 90 , missingPeakFinderSimilarityLax= 0.85, 
  autoTuneMatchStringency =FALSE, quantMethod = "T")

print(Alignment$Alignment_Matrix)

# TODO
Alignment_filtered_matrix<-Alignment$Alignment_Matrix
filter<-0.5
indexKeep<- which(apply(Alignment_filtered_matrix,1,function(x) sum(!is.na(x))>filter*ncol(Alignment_filtered_matrix)))


Alignment_filtered_matrix<-Alignment_filtered_matrix[indexKeep,]
# print(Alignment_filtered_matrix)



output_dir <- "C:/Users/adeli/Documents/programmation/uvsq/Python-2DGC-Alignment/consensus/"
write.table(Alignment$Alignment_Matrix,
            file = file.path(output_dir, "R_Alignment_Matrix.txt"),
            sep = "\t", row.names = TRUE, quote = FALSE)

write.table(Alignment$Peak_Info,
            file = file.path(output_dir, "R_Peak_Info.txt"),
            sep = "\t", row.names = FALSE, quote = FALSE)

write.table(Alignment$RT_group,
            file = file.path(output_dir, "R_RT_Group.txt"),
            sep = "\t", row.names = TRUE, quote = FALSE)

write.table(Alignment$spectra_group,
            file = file.path(output_dir, "R_Spectra_Group.txt"),
            sep = "\t", row.names = TRUE, quote = FALSE)

write.table(Alignment_filtered_matrix,
            file = file.path(output_dir, "R_Alignment_Matrix_after_filter.txt"),
            sep = "\t", row.names = TRUE, quote = FALSE)
