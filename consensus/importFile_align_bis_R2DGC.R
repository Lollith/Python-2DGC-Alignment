# library(R2DGC)

# result <- ImportFile("D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/A-F-028-817822-droite-ReCIV.txt")
# ImportFile <- function(File) { 
#   MissingStandards <- c()
#   currentRawFile <- read.table(File, sep = "\t", fill = T,  #pour .txt
#                                quote = "", strip.white = T, stringsAsFactors = F, 
#                                header = T)
#   currentRawFile[, 5] <- as.character(currentRawFile[, 
#                                                      5])
#   currentRawFile[, 2] <- as.character(currentRawFile[, 
#                                                      2])
# #filtrage des lignes  avec valeur manquantes
#   currentRawFile <- currentRawFile[which(!is.na(currentRawFile[,3]) & 
#                                         nchar(currentRawFile[,5]) != 0),]


#     # DIAGNOSTIC : Regarder quelques exemples de spectres bruts
#   cat("=== DIAGNOSTIC ===\n")
#   cat("Exemples de spectres bruts (colonne 5):\n")
#   for(i in 1:min(3, nrow(currentRawFile))) {
#     cat("Ligne", i, ":", substr(currentRawFile[i, 5], 1, 100), "...\n")
#   }

#   RTSplit <- data.frame(strsplit(currentRawFile[, 2], " , "), 
#                         stringsAsFactors = F)
#   RTSplit[1, ] <- gsub("\"", "", RTSplit[1, ])
#   RTSplit[2, ] <- gsub("\"", "", RTSplit[2, ])
#   currentRawFile[, "RT1"] <- as.numeric(t(RTSplit[1, ]))
#   currentRawFile[, "RT2"] <- as.numeric(t(RTSplit[2, ]))

#   uniqueIndex <- data.frame(paste(currentRawFile[, 1], 
#                                   currentRawFile[, 2],
#                                   currentRawFile[, 3]))
#   currentRawFile <- currentRawFile[which(!duplicated(uniqueIndex)), 
#   ]
#   row.names(currentRawFile) <- c(1:nrow(currentRawFile))

#    # Parsing des spectres avec diagnostic


#   currentRawFileSplit <- split(currentRawFile, 1:nrow(currentRawFile))
#   spectraSplit <- lapply(currentRawFileSplit, function(a) strsplit(a[[5]], " "))
#   cat("\nAprès split par espaces (premier spectre, premiers éléments):\n")
#   print(spectraSplit[[1]][[1]][1:min(5, length(spectraSplit[[1]][[1]]))])

#   spectraSplit <- lapply(spectraSplit, function(b) lapply(b, function(c) strsplit(c, ":")))
#   cat("\nAprès split par ':' (premier spectre, premiers éléments):\n")
#   print(spectraSplit[[1]][[1]][1:min(3, length(spectraSplit[[1]][[1]]))])

#   spectraSplit <- lapply(spectraSplit, function(d) t(matrix(unlist(d), nrow = 2)))
#    # Vérifier si des éléments sont NULL
#   null_elements <- which(sapply(spectraSplit, is.null))
#   if(length(null_elements) > 0) {
#     cat("Éléments NULL détectés aux positions:", null_elements, "\n")
#   }
#   # Filtrer les NULL
#   spectraSplit <- spectraSplit[!sapply(spectraSplit, is.null)]
#   if(length(spectraSplit) > 0) {
#     cat("\nDimensions de la première matrice:", dim(spectraSplit[[1]]), "\n")
#     cat("Premiers éléments de la première matrice:\n")
#     print(spectraSplit[[1]][1:min(5, nrow(spectraSplit[[1]])), ])

#     spectraSplit <- lapply(spectraSplit, function(d) apply(d, 2, as.numeric))
#     print(spectraSplit[[1]][,1])
  
#     # Filtrer à nouveau les NULL après conversion numérique
#     spectraSplit <- spectraSplit[!sapply(spectraSplit, is.null)]
#     if(length(spectraSplit) > 0) {
#       cat("\nAprès conversion numérique (premiers m/z du premier spectre):\n")
#       print(spectraSplit[[1]][1:min(10, nrow(spectraSplit[[1]])), 1])
#       # Statistiques sur les m/z du premier spectre
#       mz_values <- spectraSplit[[1]][, 1]
#       cat("\nStatistiques m/z premier spectre:\n")
#       cat("Min:", min(mz_values, na.rm = T), "\n")
#       cat("Max:", max(mz_values, na.rm = T), "\n")
#       cat("Nombre de valeurs:", length(mz_values), "\n")
#       cat("Nombre de NA:", sum(is.na(mz_values)), "\n")

#       ionNames <- spectraSplit[[1]][order(spectraSplit[[1]][, 1]), 1]
#       cat("\nPremiers ionNames après tri:\n")
#       print(ionNames[1:min(20, length(ionNames))])
#       cat("\nDerniers ionNames après tri:\n")
#       print(ionNames[max(1, length(ionNames)-19):length(ionNames)])
#   # options(scipen = 999)        
                                              
#       print(ionNames)
#       spectraSplit <- lapply(spectraSplit, function(d) d[order(d[, 1]), 2, drop = F])
#     }
#   }
#   # ionNames <- sort(unique(unlist(lapply(spectraSplit, function(d) d[, 1]))))
                                                       
#   # spectraFull <- spectraSplit                                       
#   # ionNames <- sort(unique(unlist(lapply(spectraFull, function(d) d[,1]))))                                                           
#   # spectraSplit <- lapply(spectraFull, function(d) d[order(d[, 
#   #                                                            1]), 2, drop = F])
#   # spectraSplit <- lapply(spectraSplit, function(m) t(m)) 

#   # spectraFull <- spectraSplit
#   # ionNames <- sort(unique(unlist(lapply(spectraFull, function(d) d[, 1]))))

#   # # Maintenant on garde que les intensités (triées par m/z)
#   # spectraSplit <- lapply(spectraFull, function(d) d[order(d[, 1]), 2, drop = FALSE])

#   return(list(currentRawFile, spectraSplit, MissingStandards, 
#               ionNames, spectraSplit))
#   }

# # Version avec diagnostic pour comprendre le problème
# ImportFile <- function(File) {
#   MissingStandards <- c()
  
#   # Lecture du fichier
#   currentRawFile <- read.table(File, sep = "\t", fill = T, 
#                                quote = "", strip.white = T, 
#                                stringsAsFactors = F, header = T)
#   spectres <- currentRawFile[[5]]

# # Boucle sur chaque spectre (chaîne de caractères)
#   for (i in seq_along(spectres)) {
#     elements <- unlist(strsplit(spectres[i], " "))
#     for (el in elements) {
#       parts <- unlist(strsplit(el, ":"))
#       # Vérifie que l'élément a bien deux parties non vides et convertibles en numérique
#       if (length(parts) != 2 || parts[1] == "" || parts[2] == "" ||
#           is.na(as.numeric(parts[1])) || is.na(as.numeric(parts[2]))) {
#         cat("Entrée mal formée dans la ligne", i, ":", el, "\n")
#       }
#     }
# }
  
#   # Conversion des colonnes en caractères
#   currentRawFile[, 5] <- as.character(currentRawFile[, 5])
#   currentRawFile[, 2] <- as.character(currentRawFile[, 2])
  
#   # Filtrage des lignes avec valeurs manquantes
#   currentRawFile <- currentRawFile[which(!is.na(currentRawFile[,3]) & 
#                                         nchar(currentRawFile[,5]) != 0),]
  
#   # # DIAGNOSTIC : Regarder quelques exemples de spectres bruts
#   # cat("=== DIAGNOSTIC ===\n")
#   # cat("Exemples de spectres bruts (colonne 5):\n")
#   # for(i in 1:min(3, nrow(currentRawFile))) {
#   #   cat("Ligne", i, ":", substr(currentRawFile[i, 5], 1, 100), "...\n")
#   # }
  
#   # Parsing des temps de rétention (inchangé)
#   RTSplit <- data.frame(strsplit(currentRawFile[, 2], " , "), 
#                         stringsAsFactors = F)
#   RTSplit[1, ] <- gsub("\"", "", RTSplit[1, ])
#   RTSplit[2, ] <- gsub("\"", "", RTSplit[2, ])
#   currentRawFile[, "RT1"] <- as.numeric(t(RTSplit[1, ]))
#   currentRawFile[, "RT2"] <- as.numeric(t(RTSplit[2, ]))
  
#   # Suppression des doublons (inchangé)
#   uniqueIndex <- data.frame(paste(currentRawFile[, 1],
#                                  currentRawFile[, 2], 
#                                  currentRawFile[, 3]))
#   currentRawFile <- currentRawFile[which(!duplicated(uniqueIndex)), ]
#   row.names(currentRawFile) <- c(1:nrow(currentRawFile))
  
#   # Parsing des spectres avec diagnostic
#   currentRawFileSplit <- split(currentRawFile, 1:nrow(currentRawFile))
  
#   # Étape 1: Split par espaces
#   spectraSplit <- lapply(currentRawFileSplit, function(a) strsplit(a[[5]], " "))
#   # cat("\nAprès split par espaces (premier spectre, premiers éléments):\n")
#   # print(spectraSplit[[1]][[1]][1:min(5, length(spectraSplit[[1]][[1]]))])
  
#   # Étape 2: Split par ":"
#   spectraSplit <- lapply(spectraSplit, function(b) lapply(b, function(c) strsplit(c, ":")))
#   # cat("\nAprès split par ':' (premier spectre, premiers éléments):\n")
#   # print(spectraSplit[[1]][[1]][1:min(3, length(spectraSplit[[1]][[1]]))])
  
#   # Étape 3: Conversion en matrice
#   spectraSplit <- lapply(spectraSplit, function(d) {
#     tryCatch({
#       t(matrix(unlist(d), nrow = 2))
#     }, error = function(e) {
#       cat("Erreur dans la conversion en matrice:", e$message, "\n")
#       return(NULL)
#     })
#   })
  
#   # Vérifier si des éléments sont NULL
#   null_elements <- which(sapply(spectraSplit, is.null))
#   if(length(null_elements) > 0) {
#     cat("Éléments NULL détectés aux positions:", null_elements, "\n")
#   }
  
#   # Filtrer les NULL
#   spectraSplit <- spectraSplit[!sapply(spectraSplit, is.null)]
  
#   if(length(spectraSplit) > 0) {
#     # cat("\nDimensions de la première matrice:", dim(spectraSplit[[1]]), "\n")
#     # cat("Premiers éléments de la première matrice:\n")
#     # print(spectraSplit[[1]][1:min(5, nrow(spectraSplit[[1]])), ])
    
#     # Conversion numérique
#     # spectraSplit <- lapply(spectraSplit, function(d) {
#     #   tryCatch({
#     #     apply(d, 2, as.numeric)
#     #   }, error = function(e) {
#     #     cat("Erreur dans la conversion numérique:", e$message, "\n")
#     #     return(NULL)
#     #   })
#     # })
#     spectraSplit <- lapply(spectraSplit, function(d) {
#     d <- apply(d, 2, as.numeric)
#     if (any(is.na(d[,1]))) {
#       # cat("Spectre avec NA dans m/z:\n")
#       # print(d[is.na(d[,1]), ])
#       d <- d[!is.na(d[,1]), , drop = FALSE]
#     }
#     return(d)
#   })
    
#     # Filtrer à nouveau les NULL après conversion numérique
#     spectraSplit <- spectraSplit[!sapply(spectraSplit, is.null)]
    
#     if(length(spectraSplit) > 0) {
#       cat("\nAprès conversion numérique (premiers m/z du premier spectre):\n")
#       print(spectraSplit[[1]][1:min(10, nrow(spectraSplit[[1]])), 1])
      
#       # Statistiques sur les m/z du premier spectre
#       mz_values <- spectraSplit[[1]][, 1]
#       cat("\nStatistiques m/z premier spectre:\n")
#       cat("Min:", min(mz_values, na.rm = T), "\n")
#       cat("Max:", max(mz_values, na.rm = T), "\n")
#       cat("Nombre de valeurs:", length(mz_values), "\n")
#       cat("Nombre de NA:", sum(is.na(mz_values)), "\n")
#       # Noms des ions (m/z triés)
#       # spectraSplit_clean <- lapply(spectraSplit_num, function(d) d[!is.na(d[,1]) & !is.na(d[,2]), ]) 
#       # spectraSplit_sorted <- lapply(spectraSplit_clean, function(d) d[order(d[, 1]), 2, drop = FALSE])
#       # ionNames <- spectraSplit_clean[[1]][order(spectraSplit_clean[[1]][, 1]), 1]
#       ionNames <- spectraSplit[[1]][order(spectraSplit[[1]][, 1]), 1]
      
#       cat("\nPremiers ionNames après tri:\n")
#       print(ionNames[1:min(20, length(ionNames))])
#       cat("\nDerniers ionNames après tri:\n")
#       print(ionNames[max(1, length(ionNames)-19):length(ionNames)])
      
#       # Réorganisation des spectres par ordre croissant de m/z
#       spectraSplit <- lapply(spectraSplit, function(d) d[order(d[, 1]), 2, drop = F])
      
#       return(list(currentRawFile, spectraSplit, MissingStandards, ionNames))
#     }
#   }
  
#   cat("Erreur: Impossible de parser les spectres\n")
#   return(NULL)
# }

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


fichier <- "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/A-F-028-817822-droite-ReCIV.txt"
res <- ImportFile(fichier)
to_save <- list(
  main_table = res[[1]],
  spectra_split = res[[2]],
  missing_standards = res[[3]],
  ion_names = res[[4]]
)
str(res)
# str(to_save[["ion_names"]])
ionNames<- res[[1]][, "Spectra"]
head(ionNames)
unique_ions <- sort(unique(as.numeric(ionNames)))
print(unique_ions)
write.csv(res[[1]], "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/r_current_raw_file.csv", row.names = F)
saveRDS(to_save,file="D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/r_spectra_split.rds")
write.csv(res[[4]], file = "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/r_ion_names.csv", row.names = FALSE)


# write.csv(data.frame(ion = res[[1]]$Spectra), "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/r_ion_names.csv", row.names = F)
# ions_char <- sapply(res[[4]], function(x) paste(x, collapse = " "))
# write.csv(data.frame(ion = ions_char), "D:/Dossiers Persos/Adeline/Python-2DGC-Alignment/consensus/r_ion_names.csv", row.names = FALSE)


