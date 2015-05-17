# -*- coding: utf-8 -*-
import pandas.io.data as web
import datetime
start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2013, 1, 27)
f=web.DataReader("F", 'yahoo', start, end)
f=f[["Open","High","Low","Close"]]



    #***********************************************
def sigmoid(x,type="n"):
    e=2.718281828459045
    res=1/(1+e**-x)
    if type=="n":
        return res
    elif type=="d":
        return res*(1-res)

def tangenteH(x,type="n"):
    e=2.718281828459045
    denominateur=(e**x)+(e**-x)
    res=( (e**x)-(e**-x) )/denominateur
    if type=="n":
        return res
    elif type=="d":
        return 4/denominateur**2
    #***********************************************
def old_creerArchitecture(liste):
    # on initialise un reseau
    reseau={}
    # on regarde combien de couche cache il y a + la sortie
    for numeroCouche in range(len(architecture)):
        # on creer une couche dans le reseau
        reseau[numeroCouche+1]={}
        reseau[numeroCouche+1]["weight"]=None
        # on regarde combien il y a de neurones dans cette couche
        lesNeurones=range(architecture[numeroCouche]+1)[1:]
        # pour chaque neurone, on initialise les variables
        # necessaire pour les calculs
        for neurone in lesNeurones:
            reseau[numeroCouche+1][neurone]={"potentiel":None
                                            ,"signal":None
                                            ,"erreur":None}
    return reseau
    #************************************************

def creerArchitecture(architecture,input_shape):
    import copy
    w_size=copy.copy(architecture)
    w_size.insert(0,input_shape)
    reseau={}
    for numeroCouche in range(len(architecture)):
        reseau[numeroCouche]={}
        reseau[numeroCouche]["parms"]=pd.DataFrame(columns=["potentiel","signal","erreur"])
        colonne=w_size[numeroCouche]
        ligne=architecture[numeroCouche]
        reseau[numeroCouche]["weight"]=np.random.standard_normal(size=(ligne,colonne))
    return reseau
#****************************************************************

# poid
a=np.array([0.1,0.15,0.05])
b=np.array([0.12,0.18,0.08])
reseau[1]["weight"]=np.vstack((a,b))

a=np.array([0.1,0.14])
b=np.array([0.125,0.21])
c=np.array([0.13,0.07])
reseau[2]["weight"]=np.vstack((a,b,c))

input=np.array([0.9,0.1,0.9])


# ici les inputs ne st pas les memes
for cpt,i in enumerate(w1):
    potentiel=(input*i).sum()
    reseau[1][cpt+1]["potentiel"]=potentiel
    reseau[1][cpt+1]["signal"]=sigmoid(potentiel)

for cpt,i in enumerate(w2):
    potentiel=(input*i).sum()
    reseau[2][cpt+1]["potentiel"]=potentiel
    reseau[2][cpt+1]["signal"]=sigmoid(potentiel)


# on remplace les poids pr l'exercice
reseau[0]["weight"]=w1
reseau[1]["weight"]=w2

# on initilise le reseau
architecture=[2,3]
input_shape=3#f.shape[1]
reseau=creerArchitecture(architecture,input_shape)



sum(input*reseau[0]["weight"][0])
reseau[0]["parms"]["potentiel"].append(0.15)


