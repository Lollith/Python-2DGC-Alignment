
ImportFile<-function(File){

    #Read in file
    currentRawFile<-read.table(File, sep="\t", fill=T, quote="",strip.white = T, stringsAsFactors = F,header=T)
    currentRawFile[,5]<-as.character(currentRawFile[,5])
    currentRawFile<-currentRawFile[which(!is.na(currentRawFile[,3])&nchar(currentRawFile[,5])!=0),]
    currentRawFile[,2]<-as.character(currentRawFile[,2])

    #Parse retention times
    RTSplit<-data.frame(strsplit(currentRawFile[,2], " , "), stringsAsFactors = F)
    RTSplit[1,]<-gsub("\"", "", RTSplit[1,])
    RTSplit[2,]<-gsub("\"", "", RTSplit[2,])
    currentRawFile[,"RT1"]<-as.numeric(t(RTSplit[1,]))
    currentRawFile[,"RT2"]<-as.numeric(t(RTSplit[2,]))

    #Remove identical metabolite rows
    uniqueIndex<-data.frame(paste(currentRawFile[,1], currentRawFile[,2], currentRawFile[,3]))
    currentRawFile<-currentRawFile[which(!duplicated(uniqueIndex)),]
    row.names(currentRawFile)<-c(1:nrow(currentRawFile))


    #Parse metabolite spectra into a list
    currentRawFileSplit<-split(currentRawFile,1:nrow(currentRawFile))
    spectraSplit<-lapply(currentRawFileSplit, function(a) strsplit(a[[5]]," "))

    spectraSplit<-lapply(spectraSplit, function(b) lapply(b, function(c) strsplit(c,":")))
    spectraSplit<-lapply(spectraSplit, function(d) t(matrix(unlist(d),nrow=2)))
    # spectraSplit<-lapply(spectraSplit, function(d) d[which(!d[,1]%in%commonIons),])
    spectraSplit<-lapply(spectraSplit, function(d) apply(d,2,as.numeric))
    ionNames<-spectraSplit[[1]][order(spectraSplit[[1]][,1]),1]
    spectraSplit<-lapply(spectraSplit, function(d) d[order(d[,1]),2,drop=F])
    return(list(currentRawFile,spectraSplit, ionNames))
  }



#Calculate pair wise similarity scores between all metabolite spectras
  FindMatches<-function(Sample, RT2Penalty=10, RT1Penalty=1, similarityCutoff=95,numCores=1){
    spectraFrame<-do.call(cbind,Sample[[2]])
    spectraFrame<-t(spectraFrame)
    spectraFrame<-as.matrix(spectraFrame)/sqrt(apply((as.matrix(spectraFrame))^2,1,sum))
    SimilarityMatrix<-(spectraFrame %*% t(spectraFrame))*100

    #Subtract retention time difference penalties from similarity scores
    RT1Index<-matrix(unlist(lapply(Sample[[1]][,"RT1"],function(x) abs(x-Sample[[1]][,"RT1"])*RT1Penalty)),nrow=nrow(SimilarityMatrix))
    RT2Index<-matrix(unlist(lapply(Sample[[1]][,"RT2"],function(x) abs(x-Sample[[1]][,"RT2"])*RT2Penalty)),nrow=nrow(SimilarityMatrix))
    SimilarityMatrix<-SimilarityMatrix-RT1Index-RT2Index
    diag(SimilarityMatrix)<-0

    #Find metabolites to with similarity scores greater than similarityCutoff to combine
    return(apply(SimilarityMatrix,1,function(x) which(x>=similarityCutoff)))
  }


listFiles <- list("/home/camille/Documents/app/data/output/751303_v3_E3AM_5jui.txt", 
                   "/home/camille/Documents/app/data/output/751304_v1_E3AM_4jui.txt")
ImportedFiles <- lapply(listFiles, ImportFile)

for (SampNum in (seq_along(ImportedFiles))) {
    MatchList <- FindMatches(ImportedFiles[[SampNum]])
    
    MatchListStr <- vapply(MatchList, function(x) {
      if (length(x) == 0) return("no_match")      # cas vide
      if (is.logical(x)) return(as.character(x))  # cas TRUE/FALSE
      paste(x, collapse = ",")                    # cas normal
    }, character(1))


write.table(
  data.frame(MatchListStr, stringsAsFactors = FALSE),
  file = paste0("/home/camille/Documents/app/data/output/", "R_MatchList_", SampNum, ".txt"),
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)


}


