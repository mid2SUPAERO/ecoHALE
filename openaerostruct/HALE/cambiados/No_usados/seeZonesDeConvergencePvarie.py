# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:40:13 2019

@author: e.duriez
"""

from openmdao.api import Problem, ScipyOptimizeDriver, SqliteRecorder, CaseReader
import matplotlib.pyplot as plt


resuWeightP=[]
resuMrhoP=[]
resuCO2P=[]
resuIteP=[]
resuEcart1P=[]
resuEcart2P=[]
resuSpanP=[]
resuTaperP=[]
resuChordP=[]
resuSurfP=[]
resuPowerP=[]
resuFailureP=[]
resuBucklingP=[]
resuWeightPFix=[]
resuMrhoPFix=[]
resuCO2PFix=[]
resuItePFix=[]
resuEcart1PFix=[]
resuEcart2PFix=[]
resuSpanPFix=[]
resuTaperPFix=[]
resuChordPFix=[]
resuSurfPFix=[]
resuPowerPFix=[]
resuFailurePFix=[]
resuBucklingPFix=[]

#rangeP1=range(0,11,1)
#rangeP=[1+x/10 for x in rangeP1]

rangeP=[1]
#rangeM=range(1565,8200,30)
#rangeM=range(1565,2705,30)
rangeM=mrhoiPFix

#rangeM=range(1565,8200,30)

for puissanceMM in rangeP:
    mrhoInit=rangeM
    #mrhoInit=[2850]
    resuWeight=[]
    resuMrho=[]
    resuCO2=[]
    resuIte=[]
    resuEcart1=[]
    resuEcart2=[]
    resuSpan=[]
    resuTaper=[]
    resuChord=[]
    resuSurf=[]
    resuPower=[]
    resuFailure=[]
    resuBuckling=[]
    resuWeightFix=[]
    resuMrhoFix=[]
    resuCO2Fix=[]
    resuIteFix=[]
    resuEcart1Fix=[]
    resuEcart2Fix=[]
    resuSpanFix=[]
    resuTaperFix=[]
    resuChordFix=[]
    resuSurfFix=[]
    resuPowerFix=[]
    resuFailureFix=[]
    resuBucklingFix=[]


#####

#####
    for mrhof in mrhoInit:
#    for mrhof2 in mrhoInit:
#        mrhof=mrhof2/50
        limbasserho=55
        limhauterho=8220

        ####### retreive material optimization cases 
        cr = CaseReader("aerostructMrhoi"+str(mrhof+1)+"p"+str(puissanceMM)+"limh"+str(limhauterho)+".db")
    
        driver_cases = cr.list_cases('driver')
    
        iterations=len(driver_cases)
        #$iterations=3
        case = cr.get_case(driver_cases[iterations-1])
        design_vars = case.get_design_vars()
        objective= case.get_objectives()
        constraints= case.get_constraints()
        resuMrho.append(design_vars['mrho'][0])
        resuWeight.append(case.outputs['wing.structural_mass'][0])
        resuCO2.append(objective['emitted_co2'][0])
        resuIte.append(iterations)
        resuTaper.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
        resuSpan.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
        resuChord.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
        resuSurf.append(resuSpan[-1]*resuChord[-1]*(1-resuTaper[-1])/2)
        resuPower.append(constraints['AS_point_0.enough_power'][0])
        resuFailure.append(constraints['AS_point_1.wing_perf.failure'][0])
        resuBuckling.append(constraints['AS_point_1.wing_perf.buckling'][0])
        
        if iterations==1:
            print('only one iteration!!')
            casebefore=case
            casebbefore=case
        elif iterations==2:
            casebefore=cr.get_case(driver_cases[iterations-2])
            casebbefore=cr.get_case(driver_cases[iterations-2])
        else:
            casebefore=cr.get_case(driver_cases[iterations-2])
            casebbefore=cr.get_case(driver_cases[iterations-3])
        objectiveb= casebefore.get_objectives()
        objectivebb= casebbefore.get_objectives()
        co2before=objectiveb['emitted_co2'][0]
        co2bbefore=objectivebb['emitted_co2'][0]
        resuEcart1.append(resuCO2[-1]-co2before)
        resuEcart2.append(resuCO2[-1]-co2bbefore)



        ###### retreive cases for function to optimize
        cr = CaseReader("aerostructMrhoi"+str(mrhof)+"p"+str(puissanceMM)+"limh"+str(mrhof)+".db")
    
        driver_cases = cr.list_cases('driver')
    
        iterations=len(driver_cases)
        #$iterations=3
        case = cr.get_case(driver_cases[iterations-1])
        design_vars = case.get_design_vars()
        objective= case.get_objectives()
        constraints= case.get_constraints()
        resuMrhoFix.append(design_vars['mrho'][0])
        resuWeightFix.append(case.outputs['wing.structural_mass'][0])
        resuCO2Fix.append(objective['emitted_co2'][0])
        resuIteFix.append(iterations)
        resuTaperFix.append(case.inputs['wing.geometry.mesh.taper.taper'][0])
        resuSpanFix.append(case.inputs['wing.geometry.mesh.stretch.span'][0])
        resuChordFix.append(case.inputs['wing.geometry.mesh.scale_x.chord'][0])
        resuSurfFix.append(resuSpanFix[-1]*resuChordFix[-1]*(1-resuTaperFix[-1])/2)
        resuPowerFix.append(constraints['AS_point_0.enough_power'][0])
        resuFailureFix.append(constraints['AS_point_1.wing_perf.failure'][0])
        resuBucklingFix.append(constraints['AS_point_1.wing_perf.buckling'][0])

        if iterations==1:
            print('only one iteration!!')
            casebefore=case
            casebbefore=case
        elif iterations==2:
            casebefore=cr.get_case(driver_cases[iterations-2])
            casebbefore=cr.get_case(driver_cases[iterations-2])
        else:
            casebefore=cr.get_case(driver_cases[iterations-2])
            casebbefore=cr.get_case(driver_cases[iterations-3])
        objectiveb= casebefore.get_objectives()
        objectivebb= casebbefore.get_objectives()
        co2before=objectiveb['emitted_co2'][0]
        co2bbefore=objectivebb['emitted_co2'][0]
        resuEcart1Fix.append(resuCO2[-1]-co2before)
        resuEcart2Fix.append(resuCO2[-1]-co2bbefore)



    resuWeightP.append(resuWeight)
    resuMrhoP.append(resuMrho)
    resuCO2P.append(resuCO2)
    resuIteP.append(resuIte)
    resuEcart1P.append(resuEcart1)
    resuEcart2P.append(resuEcart2)
    resuSpanP.append(resuSpan)
    resuTaperP.append(resuTaper)
    resuChordP.append(resuChord)
    resuPowerP.append(resuPower)
    resuFailureP.append(resuFailure)
    resuBucklingP.append(resuBuckling)
    resuWeightPFix.append(resuWeightFix)
    resuMrhoPFix.append(resuMrhoFix)
    resuCO2PFix.append(resuCO2Fix)
    resuItePFix.append(resuIteFix)
    resuEcart1PFix.append(resuEcart1Fix)
    resuEcart2PFix.append(resuEcart2Fix)
    resuSpanPFix.append(resuSpanFix)
    resuTaperPFix.append(resuTaperFix)
    resuChordPFix.append(resuChordFix)
    resuPowerPFix.append(resuPowerFix)
    resuFailurePFix.append(resuFailureFix)
    resuBucklingPFix.append(resuBucklingFix)

####get rid of iterations that didn't converge     
#newResuMrho=[]
#newResuWeight=[]
#newResuMrhoFix=[]
#newResuWeightFix=[]
#newMrhoInit=[]
#newMrhoInitFix=[]
#for n in range(0,len(resuMrho)):
#    if resuTime[n]!=0:
#        newResuMrho.append(resuMrho[n])
#        newResuWeight.append(resuWeight[n])
#        newMrhoInit.append(mrhoInit[n])
#    if resuTimeFix[n]!=0:
#        newResuMrhoFix.append(resuMrhoFix[n])
#        newResuWeightFix.append(resuWeightFix[n])
#        newMrhoInitFix.append(mrhoInit[n])

####get rid of iterations that didn't converge   
#newResuMrhoP=[]
#newResuWeightP=[]
#newResuCO2P=[]
#newResuIteP=[]
#newResuMrhoFixP=[]
#newResuWeightFixP=[]
#newResuCO2FixP=[]
#newResuIteFixP=[]
#newMrhoInitP=[]
#newMrhoInitFixP=[]
#i=-1
#for puissanceMM in rangeP:
#    i+=1
#    newResuMrho=[]
#    newResuWeight=[]
#    newResuCO2=[]
#    newResuIte=[]
#    newResuMrhoFix=[]
#    newResuWeightFix=[]
#    newResuCO2Fix=[]
#    newResuIteFix=[]
#    newMrhoInit=[]
#    newMrhoInitFix=[]
#    for n in range(0,len(resuMrho)):
##        if -0.1<=resuEcart2P[i][n]<=0.1:
#        if resuIteP[i][n]>=20:
#            newResuMrho.append(resuMrhoP[i][n])
#            newResuWeight.append(resuWeightP[i][n])
#            newResuCO2.append(resuCO2P[i][n])
#            newResuIte.append(resuIteP[i][n])            
#            newMrhoInit.append(mrhoInit[n])
##        if -0.1<=resuEcart2PFix[i][n]<=0.1:
#        if resuItePFix[i][n]>=20:
#            newResuMrhoFix.append(resuMrhoPFix[i][n])
#            newResuWeightFix.append(resuWeightPFix[i][n])
#            newResuCO2Fix.append(resuCO2PFix[i][n])
#            newResuIteFix.append(resuItePFix[i][n])
#            newMrhoInitFix.append(mrhoInit[n]) 
#    newResuMrhoP.append(newResuMrho)
#    newResuWeightP.append(newResuWeight)
#    newResuCO2P.append(newResuCO2)
#    newResuIteP.append(newResuIte)
#    newResuMrhoFixP.append(newResuMrhoFix)
#    newResuWeightFixP.append(newResuWeightFix)
#    newResuCO2FixP.append(newResuCO2Fix)
#    newResuIteFixP.append(newResuIteFix)
#    newMrhoInitP.append(newMrhoInit)
#    newMrhoInitFixP.append(newMrhoInitFix)


i=-1    
for puissanceMM in rangeP:
    i+=1
#Æfor puissanceMM in range(1,6):
    print(puissanceMM)
    print("var")
    #plt.plot(newMrhoInitP[i],newResuWeightP[i],'ob')
    plt.plot(mrhoInit,resuWeightP[i])
    plt.xlabel('mrho')
    plt.ylabel('weight')
    #plt.ylim((0,50000))
    
    plt.show()
    
    #plt.plot(newMrhoInitP[i],newResuMrhoP[i],'ob')
    plt.plot(mrhoInit,resuMrhoP[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('mrhof')
    #plt.ylim((0,6000))
    
    plt.show()
    
    #plt.plot(newMrhoInitP[i],newResuCO2P[i],'ob')
    plt.plot(mrhoInit,resuCO2P[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('co2')
    #plt.ylim((0,0.05))
    
    plt.show()

    #plt.plot(newMrhoInitP[i],newResuIteP[i],'ob')
    plt.plot(mrhoInit,resuIteP[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('number of iterations')
#    plt.ylim((0,100))
    
    plt.show()

#    #plt.plot(newMrhoInit,newResuMrho,'ob')
#    plt.plot(mrhoInit,resuEcart1P[i],'ob')
#    plt.xlabel('mrhoi')
#    plt.ylabel('co2 difference between last iteration and the one before')
##    plt.ylim((0,100))
#    
#    plt.show()
#
#    #plt.plot(newMrhoInit,newResuMrho,'ob')
#    plt.plot(mrhoInit,resuEcart2P[i],'ob')
#    plt.xlabel('mrhoi')
#    plt.ylabel('co2 difference between last iteration and two before')
##    plt.ylim((0,100))
#    
#    plt.show()
    
    #plt.plot(newMrhoInit,newResuMrho,'ob')
    plt.plot(mrhoInit,resuSpanP[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('span')
#    plt.ylim((0,100))
    
    plt.show()
    
    #plt.plot(newMrhoInit,newResuMrho,'ob')
    plt.plot(mrhoInit,resuTaperP[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('taper')
    #plt.ylim((0,0.1))
    
    plt.show()
    
    #plt.plot(newMrhoInit,newResuMrho,'ob')
    plt.plot(mrhoInit,resuChordP[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('chord')
#    plt.ylim((1,1.05))
    
    plt.show()
    


    
i=-1    
for puissanceMM in rangeP:
    i+=1
#Æfor puissanceMM in range(1,6):
    print(puissanceMM)
    print("fix")
    #plt.plot(newMrhoInitFixP[i],newResuWeightFixP[i],'ob')
    plt.plot(mrhoInit,resuWeightPFix[i],'ob')
    plt.xlabel('mrho')
    plt.ylabel('weight')
#    plt.xlim((2500,3000))
    #plt.ylim((0,3000))
    
    plt.show()
    
    #plt.plot(newMrhoInitFixP[i],newResuMrhoFixP[i],'ob')
    plt.plot(mrhoInit,resuMrhoPFix[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('mrhof')
#    plt.xlim((2500,3000))
    
    plt.show()
    
    #plt.plot(newMrhoInitFixP[i],newResuCO2FixP[i],'ob')
    plt.plot(mrhoInit,resuCO2PFix[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('co2')
    #plt.ylim((0,0.05))
    
    plt.show()
    
    #plt.plot(newMrhoInitFixP[i],newResuIteFixP[i],'ob')
    plt.plot(mrhoInit,resuItePFix[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('number of iterations')
#    plt.ylim((0,100))
    
    plt.show()
    
#    #plt.plot(newMrhoInitFix,newResuMrhoFix,'ob')
#    plt.plot(mrhoInit,resuEcart1PFix[i],'ob')
#    plt.xlabel('mrhoi')
#    plt.ylabel('co2 difference between last iteration and the one before')
##    plt.ylim((0,100))
#    
#    plt.show()
#    
#    #plt.plot(newMrhoInitFix,newResuMrhoFix,'ob')
#    plt.plot(mrhoInit,resuEcart2PFix[i],'ob')
#    plt.xlabel('mrhoi')
#    plt.ylabel('co2 difference between last iteration and two before')
##    plt.ylim((0,100))
#    
#    plt.show()
    
    #plt.plot(newMrhoInit,newResuMrho,'ob')
    plt.plot(mrhoInit,resuSpanPFix[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('span')
#    plt.ylim((0,100))
    
    plt.show()
    
    #plt.plot(newMrhoInit,newResuMrho,'ob')
    plt.plot(mrhoInit,resuTaperPFix[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('taper')
#    plt.ylim((0,100))
    
    plt.show()
    
    #plt.plot(newMrhoInit,newResuMrho,'ob')
    plt.plot(mrhoInit,resuChordPFix[i],'ob')
    plt.xlabel('mrhoi')
    plt.ylabel('chord')
#    plt.ylim((0,100))
    
    plt.show()
