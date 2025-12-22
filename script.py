import shutil
import os
import sys
 
file1 ="/data/FCP10/Donnees/MC/explo/FCP10_MLCDP_WAX647/code/CDP/IR.py"

file2 ="/data/FCP10/Donnees/MC/explo/FCP10_MLCDP_WAX647/code/CDP/ConnaissanceClient.py"
file3 ="/data/FCP10/Donnees/MC/explo/FCP10_MLCDP_WAX647/code/CDP/Segmentation.py"
file1 ="/data/FCP10/Donnees/MC/explo/FCP10_MLCDP_WAX647/code/CDP/Contact.py"
file1 ="/data/FCP10/Donnees/MC/explo/FCP10_MLCDP_WAX647/code/CDP/InteractionBqe.py"
#file2 ="/data/FCP10/Donnees/MC/explo/FCP10_MLCDP_WAX647/code/DeveloppementRelation/etudes/Modules exploratoires socio-démo/segpotdyn/modules/production_age_clustering.py"

shutil.copyfile(file1, "/mnt/FCP10DTWH/IR.py")
shutil.copyfile(file1, "/mnt/FCP10DTWH/ConnaissanceClient.py")
shutil.copyfile(file1, "/mnt/FCP10DTWH/Segmentation.py")
shutil.copyfile(file1, "/mnt/FCP10DTWH/Contact.py")
shutil.copyfile(file1, "/mnt/FCP10DTWH/InteractionBqe.py")

#shutil.copyfile(file2, "/mnt/FCP10DTWH/production_age_clustering.txt")