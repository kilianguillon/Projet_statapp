#Librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import scipy.stats as stats
import ast
from scipy.stats import invgamma,loggamma,invgauss
import matplotlib.patches as mpatches
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns

!pip install openpyxl


"""Données"""
#EMP=pd.read_csv("https://raw.githubusercontent.com/kilianguillon/Projet_statapp/main/data/EMP_deplacements_Charme.csv", sep=";", encoding='latin-1')
EMP=pd.read_excel("data/data.xlsx")
Simulation = pd.read_csv("simulation.csv")
EMP["HEURE_ARRIVEE"]=EMP["HEURE_ARRIVEE"].astype(float)


#Lire le tableau des lois de durée restée dans un lieu:
"""tableau_duree= pd.read_excel("data/lois_duree.xlsx")
tableau_duree=tableau_duree.set_index("Unnamed: 0").rename_axis("Plage horaire")
tableau_duree=tableau_duree.map(lambda x: ast.literal_eval(x))"""

#On crée les plages horraires :

EMP.loc[EMP["HEURE_ARRIVEE"].between(0, 11),"Plage_horraire"] = "00-11h"
EMP.loc[EMP["HEURE_ARRIVEE"].between(11, 14),"Plage_horraire"] = "11-14h"
EMP.loc[EMP["HEURE_ARRIVEE"].between(14, 17),"Plage_horraire"] = "14-17h"
EMP.loc[EMP["HEURE_ARRIVEE"].between(17, 24),"Plage_horraire"] = "17-00h"






"""Fonctions"""

def count_occ_pond(data, nom_var, nom_pond, taille=10):
    value = data[nom_var].unique() #cette fonction est utile pour des variables prenant un nbre de valeurs finies
    result_dict = {nom_var: [], 'Occurences pondérées': [], 'Proportion':[]} #à chaque valeur on associe la somme des occurences pondérées
    N=data[nom_pond].sum()
    for val in value :
        somme_pond = data[data[nom_var] == val][nom_pond].sum() 
        result_dict[nom_var].append(val)
        result_dict['Occurences pondérées'].append(somme_pond)
        result_dict['Proportion'].append(round(100*somme_pond/N,2))
    return pd.DataFrame(result_dict).sort_values('Occurences pondérées', ascending=False).reset_index(drop=True).head(taille)


#On définit une fonction qui selon les probabilités ci-dessus, va produire le lieu de départ de n individus
def loi_lieu_depart(n=1, p_domicile=70.49/100, p_rue=23.87/100, p_parking=3.90/100, p_entreprise=1.38/100, p_sans=0.36/100):
    matrice_initiale = [random.uniform(0, 1) for k in range(n)]
    for i in range(n):
        if matrice_initiale[i] <= p_domicile:
            matrice_initiale[i] = "Domicile"
        elif matrice_initiale[i] <= p_domicile + p_rue:
            matrice_initiale[i] = "Rue"
        elif matrice_initiale[i] <= p_domicile + p_rue + p_parking:
            matrice_initiale[i] = "Parking"
        elif matrice_initiale[i] <= p_domicile + p_rue + p_parking + p_entreprise:
            matrice_initiale[i] = "Entreprise"
        else:
            matrice_initiale[i] = "Sans"
    
    return matrice_initiale


#On va créer un e fonction liée à count_occ_pound qui pour chaque plage horraire va nous envoyer la proba de Lieu_Arrivee pour chaque Lieu_Depart
def transition_plage(plage, df=EMP):

    Lieux = ["Domicile","Rue","Entreprise","Parking","Sans"] 
    results=None
    for Lieu in Lieux:
        data = df[(df["Plage_horraire"] == plage) & (df["Lieu_Depart"] == Lieu)]
        results_temp = count_occ_pond(data, 'Lieu_Arrivee', 'POND_JOUR').iloc[:, [0, 2]]

        results_temp["Proportion"] = results_temp["Proportion"] / 100
        results_temp.rename(columns={'Proportion': 'Proba'+'_'+Lieu}, inplace=True)


        if results is None: #on fusionne toutes les proportions pour créer la matrice
            results = results_temp
        else:
            results = pd.merge(results, results_temp, on='Lieu_Arrivee')

    return results



#Maintenant on va créer une fonction à laquelle on donne un lieu de départ et une heure de départ et qui nous donne un lieu d'arrivée
#En suivant les proba obtenues ci-dessus
def calcul_lieu_arrivee(lieu_depart, heure_depart, df=EMP): #juste l'heure, les minutes ne sont pas utiles
    if heure_depart<11: #on trouve la plage horraire correspondante
        plage="00-11h"
    elif heure_depart<14 :
        plage="11-14h"
    elif heure_depart<17 :
        plage="14-17h"
    else:
        plage="17-00h"

    table=transition_plage(plage, df)[["Lieu_Arrivee", "Proba_"+lieu_depart]] #on prend les proba de déplacement pour cette plage et ce lieu de départ

    proba_parking = table.loc[table["Lieu_Arrivee"] == "Parking"].iloc[0,1]#proba d'aller dans un parking selon l'heure et le lieu de départ donnés.
    proba_domicile = table.loc[table["Lieu_Arrivee"] == "Domicile"].iloc[0,1]
    proba_rue = table.loc[table["Lieu_Arrivee"] == "Rue"].iloc[0,1]
    proba_entreprise = table.loc[table["Lieu_Arrivee"] == "Entreprise"].iloc[0,1]
    proba_sans = table.loc[table["Lieu_Arrivee"] == "Sans"].iloc[0,1]
    
    tirage=random.uniform(0, 1) #on construit la loi avec les probas obtenues
    if tirage < proba_parking:
        Lieu_Arrivee="Parking"
    elif tirage < proba_parking + proba_domicile:
        Lieu_Arrivee="Domicile"
    elif tirage < proba_parking + proba_domicile + proba_rue:
        Lieu_Arrivee="Rue"
    elif tirage < proba_parking + proba_domicile + proba_rue + proba_entreprise:
        Lieu_Arrivee="Entreprise"
    else:
        Lieu_Arrivee="Sans"
    
    return Lieu_Arrivee


def tableau_lois_dureelieu(dataset=EMP):
    ''' Fonction générant un tableau de lois de durée restée dans un lieu selon une base de données'''
    emp_df=dataset
    nb_dep_df=emp_df.groupby("IDENT_IND")["num_dep_V"].max().to_frame().rename(columns={"num_dep_V":"nb_dep"})
    emp_df=emp_df.set_index("IDENT_IND")
    emp_df["nb_dep"]=nb_dep_df["nb_dep"]
    emp_df=emp_df.reset_index()

    data=emp_df
    data["Durée"]=0 #On crée une colonne "durée"
    data.loc[data["nb_dep"]==1,"Durée"]=24-data["HEURE_ARRIVEE"]
    data.loc[data["nb_dep"]==data["num_dep_V"],"Durée"]=24-data["HEURE_ARRIVEE"]
    #On remplit la colonne "durée" en faisant la différence entre l'heure du prochain départ de l'individu et l'heure d'arriver à son lieu actuel.
    data.loc[(data["nb_dep"]!=1)&(data["nb_dep"]!=data["num_dep_V"]),'Durée'] = data.groupby('IDENT_IND')["HEURE_DEPART"].shift(-1) - data["HEURE_ARRIVEE"]

    emp_df=data
    # Matin (départ entre 00h et 11h) :
    emp_matin=emp_df[emp_df["HEURE_DEPART"]<=11]
    # Midi (départ entre 11h et 14h) :
    emp_midi=emp_df[(emp_df["HEURE_DEPART"]>11)&( emp_df["HEURE_DEPART"]<=14)]
    # Après-midi (départ entre 14h et 17h) :
    emp_am=emp_df[(emp_df["HEURE_DEPART"]>14)&( emp_df["HEURE_DEPART"]<=17)]
    # Soir (départ entre 17h et 00h) :
    emp_soir=emp_df[(emp_df["HEURE_DEPART"]>17)&( emp_df["HEURE_DEPART"]<=24)]

    dist_names = ['norm','gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss', 'chi2','beta']

    #Matin
    lois_duree_matin=[]

    for lieu in list(set(emp_df["Lieu_Arrivee"])):
        #emp_matin[emp_matin["Lieu_Arrivee"]==lieu]["Durée"].plot.hist(ax=ax[i][j],bins=20,density=True,ylabel='Fréquence',subplots=True,xlabel="Durée passée sur le lieu : "+lieu+" ")
        '''On cherche la loi qui a la plus petite somme des résidus au carré'''
        data=emp_matin[emp_matin["Lieu_Arrivee"]==lieu]["Durée"].dropna()
        sse = np.inf 
        y, x = np.histogram(data, bins=48, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Pour chaque distribution
        for name in dist_names:

            # Modéliser
            dist = getattr(stats, name)
            param = dist.fit(data)
        
            # Paramètres
            loc = param[-2]
            if abs(loc)<10**(-2):
                loc=0
            scale = param[-1]
            arg = param[:-2]

            # PDF
            pdf = dist.pdf(x, *arg, loc=loc, scale=scale)
            # SSE
            model_sse = np.sum((y - pdf)**2)

            # Si le SSE est diminué, enregistrer la loi
            if model_sse < sse:
                best_pdf = pdf
                sse = model_sse
                best_loc = loc
                best_scale = scale
                best_arg = arg
                best_param=param
                best_name = name
        lois_duree_matin.append([best_name,best_param])

        #Midi:

        lois_duree_midi=[]

    for lieu in list(set(emp_df["Lieu_Arrivee"])):
        #emp_midi[emp_midi["Lieu_Arrivee"]==lieu]["Durée"].plot.hist(ax=ax[i][j],bins=20,density=True,ylabel='Fréquence',subplots=True,xlabel="Durée passée sur le lieu : "+lieu+" ")
        '''On cherche la loi qui a la plus petite somme des résidus au carré'''
        data=emp_midi[emp_midi["Lieu_Arrivee"]==lieu]["Durée"].dropna()
        sse = np.inf 
        y, x = np.histogram(data, bins=48, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Pour chaque distribution
        for name in dist_names:

            # Modéliser
            dist = getattr(stats, name)
            param = dist.fit(data)

            # Paramètres
            loc = param[-2]
            scale = param[-1]
            arg = param[:-2]
            if abs(loc)<10**(-2):
                loc=0
            # PDF
            pdf = dist.pdf(x, *arg, loc=loc, scale=scale)
            # SSE
            model_sse = np.sum((y - pdf)**2)

            # Si le SSE est diminué, enregistrer la loi
            if model_sse < sse:
                best_pdf = pdf
                sse = model_sse
                best_loc = loc
                best_scale = scale
                best_arg = arg
                best_param=param
                best_name = name
        lois_duree_midi.append([best_name,best_param]) 

    #Après-midi:
    lois_duree_am=[]

    for lieu in list(set(emp_df["Lieu_Arrivee"])):
        #emp_am[emp_am["Lieu_Arrivee"]==lieu]["Durée"].plot.hist(ax=ax[i][j],bins=20,density=True,ylabel='Fréquence',subplots=True,xlabel="Durée passée sur le lieu : "+lieu+" ")
        '''On cherche la loi qui a la plus petite somme des résidus au carré'''
        data=emp_am[emp_am["Lieu_Arrivee"]==lieu]["Durée"].dropna()
        sse = np.inf 
        y, x = np.histogram(data, bins=48, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Pour chaque distribution
        for name in dist_names:

            # Modéliser
            dist = getattr(stats, name)
            param = dist.fit(data)

            # Paramètres
            loc = param[-2]
            scale = param[-1]
            arg = param[:-2]
            if abs(loc)<10**(-2):
                loc=0

            # PDF
            pdf = dist.pdf(x, *arg, loc=loc, scale=scale)
            # SSE
            model_sse = np.sum((y - pdf)**2)

            # Si le SSE est diminué, enregistrer la loi
            if model_sse < sse:
                best_pdf = pdf
                sse = model_sse
                best_loc = loc
                best_scale = scale
                best_arg = arg
                best_param=param
                best_name = name
    
        lois_duree_am.append([best_name,best_param])

    #Soir:
    lois_duree_soir=[]

    for lieu in list(set(emp_df["Lieu_Arrivee"])):
        #emp_soir[emp_soir["Lieu_Arrivee"]==lieu]["Durée"].plot.hist(ax=ax[i][j],bins=20,density=True,ylabel='Fréquence',subplots=True,xlabel="Durée passée sur le lieu : "+lieu+" ")
        '''On cherche la loi qui a la plus petite somme des résidus au carré'''
        data=emp_soir[emp_soir["Lieu_Arrivee"]==lieu]["Durée"].dropna()
        sse = np.inf 
        y, x = np.histogram(data, bins=48, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        # Pour chaque distribution
        for name in dist_names:

            # Modéliser
            dist = getattr(stats, name)
            param = dist.fit(data)

            # Paramètres
            loc = param[-2]
            scale = param[-1]
            arg = param[:-2]
            if abs(loc)<10**(-2):
                loc=0

            # PDF
            pdf = dist.pdf(x, *arg, loc=loc, scale=scale)
            # SSE
            model_sse = np.sum((y - pdf)**2)

            # Si le SSE est diminué, enregistrer la loi
            if model_sse < sse :
                best_pdf = pdf
                sse = model_sse
                best_loc = loc
                best_scale = scale
                best_arg = arg
                best_param=param
                best_name = name
        lois_duree_soir.append([best_name,best_param])


    #Création du tableau des lois :
    lois_duree_df=pd.DataFrame(index=["Matin","Midi","Après-midi","Soir"],columns=list(set(emp_df["Lieu_Arrivee"])))
    lois_duree_df.loc["Matin"]=lois_duree_matin
    lois_duree_df.loc["Midi"]=lois_duree_midi
    lois_duree_df.loc["Après-midi"]=lois_duree_am
    lois_duree_df.loc["Soir"]=lois_duree_soir
    return lois_duree_df



def duree_lieu(heure_arrivee,lieu_arrivee,dataset=EMP):
    '''Fonction durée restée dans un lieu: prend une heure d'arrivée en format numérique et un lieu,ainsi qu'un dataset et renvoie une durée'''
    i=0
    while i==0:
        if heure_arrivee<=11:
            plage= "Matin"
        elif heure_arrivee<=14:
            plage="Midi"
        elif heure_arrivee<=17:
            plage= "Après-midi"
        else:
            plage= "Soir"
        loi=tableau_lois_dureelieu(dataset).loc[plage,lieu_arrivee]
        dist=loi[0]
        #print(dist)
        param=list(loi[1])
        if abs(param[-2])<10**(-2):
            param[-2]=0
        #print(param)
        sample = getattr(stats,dist).rvs(*param)
        if sample>=0:
            i=1
    return round(sample,2)



Lois = [['lognorm', (0.8315232196253889, -0.04693502243267075, 13.901157095818125)], ['lognorm', (0.740185046920288, -0.25589461039479644, 12.666298036635478)], ['lognorm', (0.765396234015201, -0.3988473702700563, 15.391866767198913)],['invgauss', (0.5578367640441486, -2.0335629864492857, 37.452789564587334)]]

def duree_trajet(heure_depart):
    if heure_depart<=11:
        loi = Lois[0]
        sample = np.random.lognormal(mean=np.log(loi[1][2]), sigma=loi[1][0]) + loi[1][1]
    elif heure_depart<=14:
        loi = Lois[1]
        sample = np.random.lognormal(mean=np.log(loi[1][2]), sigma=loi[1][0]) + loi[1][1]
    elif heure_depart<=17:
        loi = Lois[2]
        sample = np.random.lognormal(mean=np.log(loi[1][2]), sigma=loi[1][0]) + loi[1][1]
    else:
        loi = Lois[3]
        sample = invgauss.rvs(mu=loi[1][0], loc=loi[1][1], scale=loi[1][2])
    return round(sample/60,2)











#Fonction finale pour notre première simulation (elle doit sortir la journée de déplacements d'un individu).
#On reprend le format de la base EMP ? (une ligne = un déplacement)

from scipy.stats import lognorm

def simule_premier_depart():
    loc = -0.6029
    scale = 10.4792
    s = 0.3005  # s est l'écart-type de la distribution normale sous-jacente
    # Génère un échantillon à partir de la loi lognormale
    sample = lognorm(s=s, scale=np.exp(loc)).rvs(size=1) * scale
    return sample[0]


#c'est un schéma d'algo pour l'instant
def simulation(data=EMP,n=1): #n le nombre d'individu que l'on simule
    
    Jour=[] #nom de la table que l'on va compléter pour tous les individus

    for individu in range(n):
        
        trajet_realise=0 
        lieu_depart = loi_lieu_depart(1)[0]
        temps_attente=simule_premier_depart()
        temps_trajet=duree_trajet(temps_attente)
        heure_arrivee=temps_attente+temps_trajet #heure d'arrivée premier trajet
        
        while heure_arrivee<24: # (on vérifie à chaque fois qu'on n'a pas fini la journée)
            if trajet_realise == 0: #c'est le premier déplacement
                trajet_realise=1
                lieu_arrivee = calcul_lieu_arrivee(lieu_depart,heure_arrivee,data)
                Jour.append([individu,lieu_depart,temps_attente, temps_trajet, temps_attente, heure_arrivee, lieu_arrivee,trajet_realise]) 
                #on implémente le lieu de départ, d'arrivée, le temps d'attente et de trajet que l'on a calaculé précedemment
                #on calcule déjà l'heure d'arrivée pour savoir si on a dépassé les 24 heures
                lieu_depart = lieu_arrivee  #lieu arrivee du déplacement précédent
                temps_attente = duree_lieu(heure_arrivee,lieu_depart)    
                heure_depart = heure_arrivee + temps_attente
                temps_trajet= duree_trajet(heure_depart)    #Solène et Guilhem parts
                heure_arrivee =heure_depart+temps_trajet

            else : #à partir du second trajet
                trajet_realise += 1
                lieu_arrivee = calcul_lieu_arrivee(lieu_depart,heure_arrivee, data)
                Jour.append([individu, lieu_depart,temps_attente, temps_trajet,heure_depart, heure_arrivee, lieu_arrivee, trajet_realise])
                lieu_depart= lieu_arrivee  #lieu arrivee du déplacement précédent
                temps_attente= duree_lieu(heure_arrivee,lieu_depart)        #Solène et Guilhem parts
                heure_depart = heure_arrivee+temps_attente #heure de départ du procahin
                temps_trajet=duree_trajet(heure_depart)
                heure_arrivee += temps_attente+temps_trajet
   
    
    Jour = pd.DataFrame(Jour, columns=["Individu","Lieu_depart","Temps_attente","Temps_trajet","Heure_depart","Heure_arrivee","Lieu_arrivee","Numero_trajet"])
    return Jour.sort_values(by='Heure_arrivee', ascending=True).reset_index(drop=True)





def plot_individual_travels_final(travel_data):
    """
    Adapted version to dynamically handle any parking location types and plot the daily travels of an individual,
    including true initial and final parking information with dynamic color coding.
    
    Parameters:
    - travel_data: DataFrame containing the travel information for an individual, including departure and arrival times,
      parking locations, departure locations, trip number, and parking information.
    """
    # Ensure the data is sorted by departure time
    travel_data_sorted = travel_data.sort_values(by='Heure_depart')
    
    # Start plotting
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Dynamically create a color map for parking locations
    parking_locations = pd.concat([travel_data_sorted['Lieu_depart'], travel_data_sorted['Lieu_arrivee']]).unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(parking_locations)))
    color_map = {location: color for location, color in zip(parking_locations, colors)}
    
    # Plot initial parking segment (from midnight to first departure)
    first_departure = travel_data_sorted.iloc[0]['Heure_depart']
    initial_parking_color = color_map[travel_data_sorted.iloc[0]['Lieu_depart']]
    ax.plot([0, first_departure], [1, 1], color=initial_parking_color, linewidth=8)
    
    # Loop through each trip to plot
    for index, row in travel_data_sorted.iterrows():
        start = row['Heure_depart']
        end = row['Heure_arrivee']
        parking_location = row['Lieu_arrivee']
        
        # Plot the travel segment
        ax.plot([start, end], [1, 1], color='black', linewidth=8)  # Uniform line thickness for travel
        
        # Plot the parking segment with slight spacing
        if index < len(travel_data_sorted) - 1:
            next_start = travel_data_sorted.iloc[index + 1]['Heure_depart']
            parking_color = color_map[parking_location]
            ax.plot([end, next_start], [1, 1], color=parking_color, linewidth=8)
    
    # Plot final parking segment (from last arrival to midnight)
    last_arrival = travel_data_sorted.iloc[-1]['Heure_arrivee']
    final_parking_color = color_map[travel_data_sorted.iloc[-1]['Lieu_arrivee']]
    ax.plot([last_arrival, 24], [1, 1], color=final_parking_color, linewidth=8)
    
    # Improving the plot aesthetics
    ax.set_xlim(0, 24)  # Set x-axis to span from midnight to midnight
    ax.set_yticks([])  # Hide y-axis as it's not relevant
    ax.set_xlabel("Heure")
    plt.title("Déplacements journaliers d'un individu avec stationnements dynamiques")
    # Create legend entries for parking locations
    legend_entries = [mpatches.Patch(color=color, label=label) for label, color in color_map.items()]
    # Add legend to the plot
    ax.legend(handles=legend_entries, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()





#Fonction analysant le dataset de training pour créer un modèle de la vitesse, afin de prédire la vitesse de nos emplacements (but final : estimer la conso électrique)
def coefvitesse(data, test_data): #on prend le sample EMP qui nous intéresse

    data["HEURE_DEPART"]=data["HEURE_DEPART"].astype(float)
    data = data.rename(columns={'HEURE_ARRIVEE': 'Heure_arrivee'})
    data = data.rename(columns={'HEURE_DEPART': 'Heure_depart'})
    data["Temps_trajet"]=data["Heure_arrivee"]-data["Heure_depart"]
    data["DISTANCE"]=data["DISTANCE"].astype(float)
    data["VITESSE"]=data["DISTANCE"]/data["Temps_trajet"]
    train_data=data[["Temps_trajet", 'Heure_depart',"VITESSE"]].dropna()
    
    # Séparez les variables explicatives (X) et la variable cible (y)
    X = train_data[["Temps_trajet", 'Heure_depart']]
    y = train_data['VITESSE']
    
# Créez le modèle de régression linéaire
    model = sm.OLS(y, sm.add_constant(X))
    
    # Ajustez le modèle aux données
    results = model.fit()

    # Préparation des données de prédiction
    pred_X = test_data[["Temps_trajet", 'Heure_depart']]
    pred_X = sm.add_constant(pred_X)
    
    # Prédiction sur les nouvelles données
    pred_y = results.predict(pred_X)
    
    # Ajout des valeurs prédites au DataFrame de prédiction
    test_data['Vitesse_predite'] = pred_y
    test_data["Distance_predite"] = test_data["Vitesse_predite"]*test_data["Temps_trajet"] #en km
    test_data["Consommation"] = 0.17*test_data["Distance_predite"] #en KWh
    
    return test_data


from scipy.stats import ttest_ind_from_stats

def compare_weighted_means_and_test_adequacy(EMP1, simu):
    # Calculer la moyenne et l'écart-type des valeurs maximum de "num_dep_V" pour EMP1 et EMP2
    mean1 = EMP1.groupby("IDENT_IND")["num_dep_V"].max().mean()
    std1 = EMP1.groupby("IDENT_IND")["num_dep_V"].max().std()
    nobs1 = EMP1.groupby("IDENT_IND")["num_dep_V"].max().count()

    mean2 = simu.groupby("Individu")["Numero_trajet"].max().mean()
    std2 = simu.groupby("Individu")["Numero_trajet"].max().std()
    nobs2 = simu.groupby("Individu")["Numero_trajet"].max().count()

    print(f"Moyenne de num_dep_V pour EMP : {mean1}")
    print(f"Moyenne de num_dep_V pour nos simulations : {mean2}")

    # Effectuer un test d'adéquation pondéré (test t de Student indépendant à partir des statistiques)
    t_stat, p_value = ttest_ind_from_stats(mean1=mean1, std1=std1, nobs1=nobs1,
                                           mean2=mean2, std2=std2, nobs2=nobs2)

    print(f"Statistique de test t : {t_stat}")
    print(f"Valeur p : {p_value}")

    # Interpréter le résultat du test
    alpha = 0.05
    if p_value < alpha:
        print("Les moyennes sont statistiquement différentes.")
    else:
        print("Les moyennes sont statistiquement similaires.")


#On crée une fonction qui va nous calculer le temps moyen passé dans chaque lieu de stationement de notre base de données
#Le but étant de comparer EMP et nos simulations
def parking(data):
    # Trier le DataFrame par 'IDENT_IND' et 'heure_depart'
    data = data.sort_values(by=['IDENT_IND', 'HEURE_DEPART'])
    
    # Calculer le temps d'attente pour chaque trajet
    data['temps_attente'] = data.groupby('IDENT_IND')['HEURE_DEPART'].diff().fillna(data["HEURE_DEPART"])
    
    # Identifier le dernier trajet de la journée pour chaque individu
    data['dernier_trajet_journee'] = data.groupby('IDENT_IND')['HEURE_ARRIVEE'].transform('max')
    
    # Calculer le temps d'attente jusqu'à minuit pour le dernier trajet de chaque journée
    data['temps_attente_jusqua_minuit'] = 24 - data['dernier_trajet_journee']
    
    # Mettre à 0 le temps d'attente jusqu'à minuit lorsque heure_arrivee est différent de dernier_trajet_journee
    data.loc[EMP['HEURE_ARRIVEE'] != data['dernier_trajet_journee'], 'temps_attente_jusqua_minuit'] = 0

    # Regrouper les données par 'IDENT_IND' et 'Lieu_Depart' et sommer les temps d'attente
    EMP_ATT = data.groupby(['IDENT_IND', 'Lieu_Depart', 'Lieu_Arrivee'])[['temps_attente', 'temps_attente_jusqua_minuit']].sum().reset_index()
    
    # Sélectionner les colonnes nécessaires pour EMP_ATT1 et renommer les colonnes
    EMP_ATT1 = EMP_ATT[["IDENT_IND", "Lieu_Depart", "temps_attente"]].copy()
    EMP_ATT1.rename(columns={
        'IDENT_IND': 'Identifiant',
        'Lieu_Depart': 'Lieu',
        'temps_attente': 'TempsAttente'
    }, inplace=True)
    
    # Sélectionner les colonnes nécessaires pour EMP_ATT2, ajuster la colonne temps_attente_jusqua_minuit et renommer la colonne
    EMP_ATT2 = EMP_ATT[["IDENT_IND", "Lieu_Arrivee", "temps_attente_jusqua_minuit"]].copy()
    EMP_ATT2.rename(columns={
        'IDENT_IND': 'Identifiant',
        'Lieu_Arrivee': 'Lieu',
        'temps_attente_jusqua_minuit': 'TempsAttente'
    }, inplace=True)
    
    # Concaténer les DataFrames EMP_ATT1 et EMP_ATT2, puis regrouper par 'Identifiant' et 'Lieu' et sommer les temps d'attente
    EMP_ATT = pd.concat([EMP_ATT1, EMP_ATT2], ignore_index=True).groupby(['Identifiant', 'Lieu'])["TempsAttente"].sum().reset_index() 

    # Créer une table pivot avec les lieux comme colonnes et les individus comme index
    #Comme ça on a 0 lorsque l'individu n'attend pas dans ce lieu
    EMP_ATT_p=EMP_ATT.pivot_table(index='Identifiant', columns='Lieu', values='TempsAttente', fill_value=0).reset_index()

    return EMP_ATT_p

def parkingsimu(data):
    # Trier le DataFrame par 'IDENT_IND' et 'heure_depart'
    data = data.sort_values(by=['Individu', 'Heure_depart'])
    
    # Calculer le temps d'attente pour chaque trajet
    data['temps_attente'] = data.groupby('Individu')['Heure_depart'].diff().fillna(data["Heure_depart"])
    
    # Identifier le dernier trajet de la journée pour chaque individu
    data['dernier_trajet_journee'] = data.groupby('Individu')['Heure_depart'].transform('max')
    
    # Calculer le temps d'attente jusqu'à minuit pour le dernier trajet de chaque journée
    data['temps_attente_jusqua_minuit'] = 24 - data['dernier_trajet_journee']
    
    # Mettre à 0 le temps d'attente jusqu'à minuit lorsque heure_arrivee est différent de dernier_trajet_journee
    data.loc[data['Heure_depart'] != data['dernier_trajet_journee'], 'temps_attente_jusqua_minuit'] = 0

    # Regrouper les données par 'IDENT_IND' et 'Lieu_Depart' et sommer les temps d'attente
    EMP_ATT = data.groupby(['Individu', 'Lieu_depart', 'Lieu_arrivee'])[['temps_attente', 'temps_attente_jusqua_minuit']].sum().reset_index()
    
    # Sélectionner les colonnes nécessaires pour EMP_ATT1 et renommer les colonnes
    EMP_ATT1 = EMP_ATT[["Individu", "Lieu_depart", "temps_attente"]].copy()
    EMP_ATT1.rename(columns={
        'Individu': 'Identifiant',
        'Lieu_depart': 'Lieu',
        'temps_attente': 'TempsAttente'
    }, inplace=True)
    
    # Sélectionner les colonnes nécessaires pour EMP_ATT2, ajuster la colonne temps_attente_jusqua_minuit et renommer la colonne
    EMP_ATT2 = EMP_ATT[["Individu", "Lieu_arrivee", "temps_attente_jusqua_minuit"]].copy()
    EMP_ATT2.rename(columns={
        'Individu': 'Identifiant',
        'Lieu_arrivee': 'Lieu',
        'temps_attente_jusqua_minuit': 'TempsAttente'
    }, inplace=True)
    
    # Concaténer les DataFrames EMP_ATT1 et EMP_ATT2, puis regrouper par 'Identifiant' et 'Lieu' et sommer les temps d'attente
    EMP_ATT = pd.concat([EMP_ATT1, EMP_ATT2], ignore_index=True).groupby(['Identifiant', 'Lieu'])["TempsAttente"].sum().reset_index() 

    # Créer une table pivot avec les lieux comme colonnes et les individus comme index
    #Comme ça on a 0 lorsque l'individu n'attend pas dans ce lieu
    EMP_ATT_p=EMP_ATT.pivot_table(index='Identifiant', columns='Lieu', values='TempsAttente', fill_value=0).reset_index()

    return EMP_ATT_p


from scipy.stats import ks_2samp

def kolmosmir(test_data, simulation):

    pivot_table1 = parking(test_data)
    pivot_table2 = parkingsimu(simulation)
    # Sélectionner les colonnes des cinq lieux
    lieux = ['Domicile', 'Rue', 'Entreprise', 'Sans', 'Parking']
    
    # Créer des DataFrames pour stocker les résultats des tests
    ks_results = pd.DataFrame(index=lieux, columns=['statistic', 'pvalue', 'nobs1', 'nobs2', 'accepte_hypothese'])

    alpha=0.05
    
    # Effectuer le test de Kolmogorov-Smirnov pour chaque lieu
    for lieu in lieux:
        # Calculer les statistiques pour les deux distributions
        ks_statistic, ks_pvalue = ks_2samp(pivot_table1[lieu], pivot_table2[lieu])
        nobs_test = len(pivot_table1[lieu])
        nobs_simu = len(pivot_table2[lieu])
            # Déterminer si l'hypothèse est acceptée ou rejetée
        accepte_hypothese = ks_pvalue > alpha
        
        # Stocker les résultats dans le DataFrame
        ks_results.loc[lieu] = [ks_statistic, ks_pvalue, nobs_test, nobs_simu, accepte_hypothese]
    
    # Afficher les résultats des tests
    return ks_results





# Simulation de 10 000 journées types pour renvoyer la consommation journalière de chaque journée

def donnees_simulees(simulation):
    '''Fonction qui prend une simulation de journées types de déplacements et calcule la consommation éléctrique associée à chaque journée en partant du principe que la consommaton d'une voiture electrique est de 17 kwh/100km'''
    simulations=simulation 
    simulations_consos=coefvitesse(EMP,simulations)
    conso_journees_types=simulations_consos.groupby(['Individu'])['Consommation'].sum().to_frame().apply(lambda x:round(x,3)).rename(columns={'Consommation':'Consommation (kwh)'}) # Renvoie un tableau de la consommation totale sur la journée de l'individu
    return conso_journees_types




'''EMP_vitesse est la base de donnée EMP avec des colonne vitesse et consommation en plus'''

EMP_vitesse= EMP.copy()
EMP_vitesse['Vitesse (km/h)']=EMP_vitesse['DISTANCE']/(EMP_vitesse['HEURE_ARRIVEE']-EMP_vitesse['HEURE_DEPART'])
EMP_vitesse ['Consommation (kwh)']= 0.17*EMP_vitesse['DISTANCE']
#EMP_vitesse.head()

def densite_conso_journaliere(donnees_simulees):
    '''Fonction qui prend en entrée une simulation effectuée par la fonction 'donnees_simulees' et renvoie la densité de l'energie consommée pour cette simulation '''
    simulations=donnees_simulees
    fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    EMP_conso= EMP_vitesse.groupby('IDENT_IND')['Consommation (kwh)'].sum().to_frame()
    sns.kdeplot(data=EMP_conso, fill=True, legend=False,ax=ax[1])
    sns.kdeplot(data=simulations, fill=True,legend=False, ax=ax[0])
    plt.suptitle("Densité de la consommation journalière d'un individu (kwh)")
    ax[0].set_xlabel('Consommation (kwh)  ,   conso moyenne = '+ str(float(round(simulations.mean(),2)))+' kwh')
    ax[1].set_xlabel('Consommation (kwh)  ,   conso moyenne = '+ str(float(round(EMP_conso.mean(),2)))+' kwh')
    ax[0].set_ylabel('Densité')
    ax[1].set_ylabel('Densité')
    ax[0].set_title("Simulation")
    ax[1].set_title("Données empiriques EMP")
    plt.show()



def quantiles_conso_journaliere(simulation):
    ''' Fonction qui prend en argument une simulation et renvoie le taleau des quantiles et exart-type de la consommation journalière d'un individu'''
    a=donnees_simulees(simulation).describe().transpose()
    b=EMP_vitesse.groupby('IDENT_IND')['Consommation (kwh)'].sum().to_frame().describe().transpose()
    return pd.concat([a,b],axis=0).set_axis(['Simulation : Consommation journalière (kwh)','EMP : Consommation journalière (kwh)'])


def heures_consommation(dataset):
    '''Prend en argument un dataset avec colonne consommation et ajoute une colonne 'Heure' contenant l'heure entière (8,9,10...) à laquelle on considère que l'energie du trajet a été consommée'''
    dataset=dataset.rename(columns=lambda x: x.upper())
    dataset['Heure'] = dataset['HEURE_ARRIVEE'].astype(int)
    colonne_conso= [colonne for colonne in dataset.columns if colonne.startswith('CONSOMMATION')]
    dataset=dataset.rename(columns={colonne_conso[0]:'Consommation (kwh)' })
    return dataset
    
def densite_conso_temps(simulation,dataset=EMP_vitesse):
    '''Fonction qui prend en argument une simulation effectuée par la fonction "simulation" une base de données EMP avec colonne consommation et affiche la densité de la consommation par rapport au temps de cette simulation ainsi que celle de la base de donnée EMP. Elle renvoie également les deux tableau du pourcentage de la consommation à chaque heure pour la simulation et EMP)'''
    '''On somme la consommation de chaque heure de la journée sur toute la base de données et on normalise'''
    #Pour simplifier, on suppose la consommation d'un trajet se fait à l'heure d'arrivée (car les trajets sont des trajets courtes distances
    simulation_consos=coefvitesse(EMP,simulation)
    fig,ax=plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
    tableau_conso=pd.DataFrame(index=list(np.arange(0,24)))
    tableau_conso['Consommation (kwh)']=0
    df=heures_consommation(simulation_consos)
    for i in tableau_conso.index:
        tableau_conso.loc[i,'Consommation (kwh)']=df.loc[df['Heure']==i]['Consommation (kwh)'].sum()
    total_consommation=tableau_conso['Consommation (kwh)'].sum()
    tableau_conso['Pourcentage simulation'] = (tableau_conso['Consommation (kwh)'] / total_consommation)
    ax[0].plot(tableau_conso.index, tableau_conso['Pourcentage simulation'], marker='o', linestyle='-', color='b')
    ax[0].fill_between(tableau_conso.index, tableau_conso['Pourcentage simulation'], color='b', alpha=0.3)
    tableau_conso2=tableau_conso.drop(['Pourcentage simulation'],axis=1).copy()
    dataset=heures_consommation(dataset)
    for i in tableau_conso2.index:
        tableau_conso2.loc[i,'Consommation (kwh)']=dataset.loc[dataset['Heure']==i]['Consommation (kwh)'].sum()
    total_consommation2=tableau_conso2['Consommation (kwh)'].sum()
    tableau_conso2['Pourcentage EMP'] = (tableau_conso2['Consommation (kwh)'] / total_consommation2)
    ax[1].plot(tableau_conso2.index, tableau_conso2['Pourcentage EMP'], marker='o', linestyle='-', color='b')
    ax[1].fill_between(tableau_conso2.index, tableau_conso2['Pourcentage EMP'], color='b', alpha=0.3)
    ax[0].set_xticks(np.arange(24))
    ax[1].set_xticks(np.arange(24))
    ax[0].set_title('Simulation')
    ax[0].set_ylabel('Densité')
    ax[0].set_xlabel('Temps (h)')
    ax[1].set_title('Base de données EMP')
    ax[1].set_ylabel('Densité')
    ax[1].set_xlabel('Temps (h)')
    plt.suptitle('Densités de consommation par rapport au temps au cours de la journée')
    plt.show()
    return pd.concat([tableau_conso.drop(['Consommation (kwh)'],axis=1),tableau_conso2.drop(['Consommation (kwh)'],axis=1)],axis=1)

def quantiles_conso_temps(simulation,dataset=EMP_vitesse):
    '''Fonction qui prend en argument une simulation ainsi qu'un dataset de type EMP avec colonne consommation et renvoie les tableaux de quantiles de la consommation d'un trajet selon les plages horaires pour la simulation et les données EMP'''
    simulation_consos=coefvitesse(EMP,simulation)
    df=heures_consommation(simulation_consos)
    
    df1=df.loc[(df['Heure']>=0)&(df['Heure']<11)].rename(columns={'Consommation (kwh)':'Consommation Trajet Matin (kwh)' })['Consommation Trajet Matin (kwh)'].describe().transpose().to_frame()
    df2=df.loc[(df['Heure']>=11)&(df['Heure']<14)].rename(columns={'Consommation (kwh)':'Consommation Trajet Midi (kwh)' })['Consommation Trajet Midi (kwh)'].describe().transpose().to_frame()
    df3=df.loc[(df['Heure']>=14)&(df['Heure']<17)].rename(columns={'Consommation (kwh)':'Consommation Trajet AM (kwh)' })['Consommation Trajet AM (kwh)'].describe().transpose().to_frame()
    df4=df.loc[(df['Heure']>=17)&(df['Heure']<24)].rename(columns={'Consommation (kwh)':'Consommation Trajet Soir (kwh)' })['Consommation Trajet Soir (kwh)'].describe().transpose().to_frame()

    df=heures_consommation(dataset)

    df11=df.loc[(df['Heure']>=0)&(df['Heure']<11)].rename(columns={'Consommation (kwh)':'Consommation Trajet Matin (kwh)' })['Consommation Trajet Matin (kwh)'].describe().transpose().to_frame()
    df22=df.loc[(df['Heure']>=11)&(df['Heure']<14)].rename(columns={'Consommation (kwh)':'Consommation Trajet Midi (kwh)' })['Consommation Trajet Midi (kwh)'].describe().transpose().to_frame()
    df33=df.loc[(df['Heure']>=14)&(df['Heure']<17)].rename(columns={'Consommation (kwh)':'Consommation Trajet AM (kwh)' })['Consommation Trajet AM (kwh)'].describe().transpose().to_frame()
    df44=df.loc[(df['Heure']>=17)&(df['Heure']<24)].rename(columns={'Consommation (kwh)':'Consommation Trajet Soir (kwh)' })['Consommation Trajet Soir (kwh)'].describe().transpose().to_frame()
    return pd.concat([pd.concat([df1,df2,df3,df4],axis=1),pd.concat([df11,df22,df33,df44],axis=1)],axis=0,keys=['Simulation','Données EMP'])

def Kolmogorov_desnite_tps(simulation):
    '''Fonction qui prend en argument une simulation et dit si la densite de la consommation par rapport au temps sur cette simulation suit la même loi que dans la base de données EMP grâce au test de Kolmogorov'''
    densites=densite_conso_temps(h)
    statistic, pvalue = stats.ks_2samp(densites['Pourcentage simulation'], densites['Pourcentage EMP'])
    # Afficher les résultats du test
    print("Statistique KS :", statistic)
    print("p-valeur :", pvalue)

    # Interpréter les résultats
    if pvalue > 0.05:
        print("Les deux échantillons suivent la même distribution.")
    else:
        print("Les deux échantillons ne suivent pas la même distribution.")



"""
# Fonction pour créer un fichier CSV à partir d'un DataFrame Pandas
def creer_fichier_csv(nom_fichier, dataframe):
    dataframe.to_csv(nom_fichier, index=False)

# Exécuter la fonction et créer un fichier CSV
if __name__ == "__main__":
    creer_fichier_csv("simulation.csv", simulation(EMP,100)) 
    print("Fichier CSV créé avec succès!")
"""

