

##################################
# DATA AND PACKAGE ADQUISITION
##################################

# Load the packages required
library(FMM)
library(circular)
library(readxl)
library(writexl)
library(writexl)
library(ggplot2)


# Set the working directory where the data (all_back_train.csv) are saved
setwd('D:/Cristina_Yolanda/Cristina_Yolanda/SGC/DatosSuyos')



# Read the data (all_back_train.csv). You provided these data
Train <- read.csv("D:/Cristina_Yolanda/Cristina_Yolanda/SGC/DatosSuyos/all_back_train.csv", header=FALSE)
TrainNa<-Train[rowSums(is.na(Train)) < 1, ] # Only row without missing data are chosen
nr<-nrow(TrainNa)



##############################
# TEMPLATE GENERATION
##############################

# Selection of 7 short segments (cuts) for each 10sec segments to generate the template.
# Each segment is formed by 61 observations, plus 16 artificial observations on each side,
# ensuring that it starts and ends at zero.
# Only 7 cuts are considered because in some cases no more are identified. In most cases, there are up to 10.
# 61 and 16 are a reasonable, but it may be changed  

nr<-25 # To simplify we only use the 25 first signal data. We force nr=25

template<-array(0,c(nr,10,93)) # template length =93 (16+61+16).  


for (ind in 1: nr){
c<-0
a<-1
b<-0
j<-0
ImaxD<-0

for (j in 1:7) {
  c<-ImaxD*(j>1)+50+ a
  b<-100+c

  a<-c
  vData<-as.numeric(t(TrainNa[ind,a:b])) # Cutting

  ImaxD<-which.max(vData)
  a1<-max(ImaxD+c-30,1)
  b1<-min(ImaxD+c+30,1000)

  dos<-c(TrainNa[ind,a1],TrainNa[ind,b1])
  wmdos<-which.max(dos)

  mini<-as.numeric(TrainNa[ind,a1])
  mfin<-as.numeric(TrainNa[ind,b1])
  lini<-abs(as.integer(as.numeric(TrainNa[ind,a1])/15))
  lfin<-abs(as.integer(TrainNa[ind,b1]/15))

  #artificial observations
  if ( mini>0 & mfin>0 & lini>0 & lfin>0) {
    cini<-seq(from = 0, to = TrainNa[ind,a1], by = lini)
    cfin<-seq(from = TrainNa[ind,b1], to = 0, by = -lfin)
  }

  if ( mini>0 & mfin<0 & lini>0 & lfin>0) {
    cini<-seq(from = 0, to = TrainNa[ind,a1], by = lini)
    cfin<-seq(from = TrainNa[ind,b1], to = 0, by = lfin)
  }

  if ( mini<0 & mfin<0 & lini>0 & lfin>0 ) {
    cini<-seq(from = 0, to = TrainNa[ind,a1], by = -lini)
    cfin<-seq(from = TrainNa[ind,b1], to = 0, by = lfin)
  }

  if ( mini<0 & mfin>0 & lini>0 & lfin>0) {
   cini<-seq(from = 0, to = TrainNa[ind,a1], by = -lini)
    cfin<-seq(from = TrainNa[ind,b1], to = 0, by = -lfin)
  }
  
  if (lini<-0) {cini<-array(0,c(15))}
  if (lfin<-0) {lfin<-array(0,c(15))}

aa<-array(TrainNa[ind,b1],c(30))
dataaux<-c(as.numeric(t(TrainNa[ind,a1:b1])),aa)


template[ind,j,1:93]<-c(cini[1:16],dataaux[1:61],cfin[1:16])
}
}


##############################################
# FMM ANALYSIS. 5 components/waves
##############################################

# Initialization


Asn<-array(0,c(nr,5))
betan<-array(0,c(nr,5))
omegan<-array(0,c(nr,5))
alphan<-array(0,c(nr,5))
Rs2n<-array(0,c(nr,5))


As<-array(0,c(nr,4))
betas<-array(0,c(nr,4))
omegas<-array(0,c(nr,4))
alphas<-array(0,c(nr,4))
Ms<-array(0,c(nr,1))
Rs2<-array(0,c(nr,4))


waves<-array(0,c(nr,4))# Wave identification from FMM 5-wave decomposition. Check number with the component plot 
ncomp<-5 # Fix the nember of component to be fitted in the FMM analysis



#Choose your ind or run loop to analyze all (nr)
for(ind in 1:nr){
#ind<-1
  vDatas<-array(0,c(93))
  vDatas<-apply(template[ind,1:7,],2,mean) # Mean data for template
  
  # Fitting the FMM using FMM R-package
  fit9 <- fitFMM(vDatas, nback=ncomp, maxiter =3 ,lengthAlphaGrid = 50,lengthOmegaGrid = 50,
                 parallelize =TRUE,showTime=TRUE,showProgress =TRUE)
  
  plotFMM(fit9,components = F, use_ggplot2 = T) # Plot the data and FMM fit
  plotFMM(fit9,components = T, use_ggplot2 = T, legendInComponentsPlot = TRUE) # Plot the FMM wave decomposition. Useful to check wave identification

  # Save the results
  Rs2n[ind,]<-fit9@R2
  omegan[ind,]<-fit9@omega
  betan[ind,]<-fit9@beta
  alphan[ind,]<-fit9@alpha
  Asn[ind,]<-fit9@A
  Ms[ind]<-fit9@M
}

####################################################
# FMM WAVE IDENTIFICATION. 
####################################################
  
# The 3 central waves are identified.
# They are labeled from right to left as 2, 1, and 3.
# A fourth wave (labelled as 4) can be located at the right end.

  

#Choose your ind or run loop to analyze all (nr)
for (ind in 1:nr){
#ind<-1   
    auxVectori <- 1:ncomp
    
    maxi<-(cos(betan[ind,1])< 0.2)*(alphan[ind,1]>6|alphan[ind,1]<0.5)
    
    if (maxi==0) { 
      Amaxi<-(alphan[ind,1:ncomp]>5.75|alphan[ind,1:ncomp]<0.7)*(cos(betan[ind,1:ncomp])< 0.2)*
        ((Rs2n[ind,1:ncomp])) #the one with R2 max and peak max
      maxi<-which.max(Amaxi)
    }
    
    
    waves[ind,1]<-auxVectori[maxi]   
    ############## 
    
    
    maxi<-9999
    # Wave 2 located before wave 1.cos(beta) is not too low.
    # Among the waves that satisfy this condition, choose the one that captures the greatest variability.
    
    auxVectori1 <- auxVectori[auxVectori != waves[ind,1]  ]
    
    
    Amaxi<-(Rs2n[ind, auxVectori1])*(cos(betan[ind,auxVectori1])> -0.25)*((alphan[waves[ind,1]]>4)*(alphan[ind,auxVectori1]<alphan[waves[ind,1]])*(alphan[ind,auxVectori1]>4)|
                                                                            (alphan[ind,auxVectori1]<2)*(alphan[ind,auxVectori1]>alphan[waves[ind,1]]) |
                                                                            (alphan[waves[ind,1]]<3)*(alphan[ind,auxVectori1]>4.5)| 
                                                                            (alphan[waves[ind,1]]<3)*(alphan[ind,auxVectori1]<2)*(alphan[ind,auxVectori1]<alphan[waves[ind,1]]))
    
    maxi<-which.max(Amaxi)
    #in case of no wave with a minimum peak:
    
    if (max(Amaxi)==0)  {Amaxi<- (Rs2n[ind, auxVectori1]) *((alphan[waves[ind,1]]>4)*(alphan[ind,auxVectori1]<alphan[waves[ind,1]])*(alphan[ind,auxVectori1]>4)|
                                                              (alphan[ind,auxVectori1]<2)*(alphan[ind,auxVectori1]>alphan[waves[ind,1]]) |
                                                              (alphan[waves[ind,1]]<3)*(alphan[ind,auxVectori1]>4.5)| 
                                                              (alphan[waves[ind,1]]<3)*(alphan[ind,auxVectori1]<2)*(alphan[ind,auxVectori1]<alphan[waves[ind,1]]))
    
    maxi<-which.max(Amaxi)
    
    } 
    waves[ind,2]<-auxVectori1[maxi]   
    if (maxi>9980) {waves[ind,2]<-10}
    
    maxi<-9999
    
    # Wave 3  located after wave 1.cos(beta) is not too low.
    # Among the waves that satisfy this condition, choose the one that captures the greatest variability.
    
    
    auxVectori2 <- auxVectori1[auxVectori1 != waves[ind,2]  ]
    
    Amaxi<- (Rs2n[ind, auxVectori2])*(cos(betan[ind,auxVectori2])>-0.25)*((alphan[waves[ind,1]]<6.3)*(alphan[ind,auxVectori2]<6.3)*(alphan[ind,auxVectori2]>alphan[waves[ind,1]]) |
                                                                            (alphan[waves[ind,1]]<6.3)*(alphan[ind,auxVectori2]<2)| 
                                                                            (alphan[waves[ind,1]]<3)*(alphan[ind,auxVectori2]<2)*(alphan[ind,auxVectori2]>alphan[waves[ind,1]]))
    
    maxi<-which.max(Amaxi)
    
    #in case of no wave with a minimum peak:
    if (max(Amaxi)==0)  {
      Amaxi<- (Rs2n[ind, auxVectori2])*((alphan[waves[ind,1]]<6.3)*(alphan[ind,auxVectori2]<6.3)*(alphan[ind,auxVectori2]>alphan[waves[ind,1]]) |
                                          (alphan[waves[ind,1]]<6.3)*(alphan[ind,auxVectori2]<2)| 
                                          (alphan[waves[ind,1]]<3)*(alphan[ind,auxVectori2]<2)*(alphan[ind,auxVectori2]>alphan[waves[ind,1]]))
      
      maxi<-which.max(Amaxi)
    } 
    
    waves[ind,3]<-auxVectori2[maxi] 
    if (maxi>9980) {waves[ind,3]<-10}
    # Wave 4   is located after wave 3.It has positive prominetn peak if beta(wave4)<-0.5.
    
    maxi<-9999
    auxVectori3 <- auxVectori2[auxVectori2 != waves[ind,3]  ]
    
    
    Amaxi<-((alphan[ind,auxVectori3]<2.5)*(alphan[ind,auxVectori3]>alphan[waves[ind,1]])|
              (alphan[ind,auxVectori3]<2.5)*(alphan[waves[ind,1]]<6.3))           
    
    maxi<-which.max(Amaxi)
    
    
    maxi<-which.max(Amaxi)
    
    
    waves[ind,4]<-auxVectori3[maxi]     
    
    if (maxi>9980) {waves[ind,4]<-10}
    
    
    
    
    omegas[ind,]<-omegan[ind,c(waves[ind,1],waves[ind,2],waves[ind,3],waves[ind,4])]
    alphas[ind,]<-alphan[ind,c(waves[ind,1],waves[ind,2],waves[ind,3],waves[ind,4])]
    betas[ind,]<-betan[ind,c(waves[ind,1],waves[ind,2],waves[ind,3],waves[ind,4])]
    As[ind,]<-Asn[ind,c(waves[ind,1],waves[ind,2],waves[ind,3],waves[ind,4])]
    Rs2[ind,]<-Rs2n[ind,c(waves[ind,1],waves[ind,2],waves[ind,3],waves[ind,4])]
}
  
#################################################
#   Feature extraction
#################################################

# Wave patient identification 
 wavePat<-cbind(1:nr,waves)
 colnames(wavePat)<-c("ID",paste0("Label ",1:4))
 

  Features<-data.frame(
    cbind(1:nr,Rs2[1:nr,1],Rs2[1:nr,2],Rs2[1:nr,3],Rs2[1:nr,4],Ms[1:nr],As[1:nr,1],As[1:nr,2],As[1:nr,3],As[1:nr,4],
          omegas[1:nr,1],omegas[1:nr,2], omegas[1:nr,3],omegas[1:nr,4],
          sin(alphas[1:nr,1]), sin(alphas[1:nr,2]), sin(alphas[1:nr,3]), sin(alphas[1:nr,4]),
          cos(alphas[1:nr,1]),cos(alphas[1:nr,2]),cos(alphas[1:nr,3]),cos(alphas[1:nr,4]),
          sin(betas[1:nr,1]), sin(betas[1:nr,2]), sin(betas[1:nr,3]),sin(betas[1:nr,4]),
          cos(betas[1:nr,1]), cos(betas[1:nr,2]), cos(betas[1:nr,3]),cos(betas[1:nr,4])
    )
  ) 
  
  colnames(Features)<-c("ID",paste0("R2_Wave",1:4),"M",paste0("A_Wave",1:4),paste0("omega_Wave",1:4),
                      paste0("sin(alpha)_Wave",1:4),paste0("cos(alpha)_Wave",1:4),
                      paste0("sin(beta)_Wave",1:4),paste0("cos(beta)_Wave",1:4))
  
  # Seve the results. Change the path
  write_xlsx(Features, path = "D:/Cristina_Yolanda/Cristina_Yolanda/SGC/DatosSuyos/Feature_FMM.xlsx",col_names = TRUE)

  
  


