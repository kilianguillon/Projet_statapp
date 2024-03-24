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
import matplotlib.pyplot as plt


!pip install openpyxl


"""Données"""
EMP=pd.read_csv("https://raw.githubusercontent.com/kilianguillon/Projet_statapp/main/data/EMP_deplacements_Charme.csv", sep=";", encoding='latin-1')
EMP["HEURE_ARRIVEE"]=EMP["HEURE_ARRIVEE"].replace(',', '.', regex=True).astype(float)


#Lire le tableau des lois de durée restée dans un lieu:
tableau_duree= pd.read_excel("data/lois_duree.xlsx")
tableau_duree=tableau_duree.set_index("Unnamed: 0").rename_axis("Plage horaire")
tableau_duree=tableau_duree.map(lambda x: ast.literal_eval(x))

#On crée les plages horraires :

EMP.loc[EMP["HEURE_ARRIVEE"].between(0, 11),"Plage_horraire"] = "00-11h"
EMP.loc[EMP["HEURE_ARRIVEE"].between(11, 14),"Plage_horraire"] = "11-14h"
EMP.loc[EMP["HEURE_ARRIVEE"].between(14, 17),"Plage_horraire"] = "14-17h"
EMP.loc[EMP["HEURE_ARRIVEE"].between(17, 24),"Plage_horraire"] = "17-00h"






"""Fonctions"""

def count_occ_pond(data, nom_var, nom_pond, taille=10):
    value = data[nom_var].unique() #cette fonction est utile pour des variables prenant un nbre de valeurs finies
    result_dict = {nom_var: [], 'Occurences pondérées': [], 'Proportion':[]} #à chaque valeur on associe la somme des occurences pondérées
    N=data[nom_pond].replace(',', '.', regex=True).astype(float).sum()
    for val in value :
        somme_pond = data[data[nom_var] == val][nom_pond].replace(',', '.', regex=True).astype(float).sum() #replace car la base utilisée est "française"
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
def transition_plage(plage):

    Lieux = ["Domicile","Rue","Entreprise","Parking","Sans"] 
    results=None
    for Lieu in Lieux:
        data = EMP[(EMP["Plage_horraire"] == plage) & (EMP["Lieu_Depart"] == Lieu)]
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
def calcul_lieu_arrivee(lieu_depart, heure_depart): #juste l'heure, les minutes ne sont pas utiles
    if heure_depart<11: #on trouve la plage horraire correspondante
        plage="00-11h"
    elif heure_depart<14 :
        plage="11-14h"
    elif heure_depart<17 :
        plage="14-17h"
    else:
        plage="17-00h"

    table=transition_plage(plage)[["Lieu_Arrivee", "Proba_"+lieu_depart]] #on prend les proba de déplacement pour cette plage et ce lieu de départ
    
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

#Fonction durée restée dans un lieu: prend une heure d'arrivée en format numérique et un lieu et renvoie une durée
def duree_lieu(heure_arrivee,lieu):
    if heure_arrivee<=11:
        plage= "Matin"
    elif heure_arrivee<=14:
        plage="Midi"
    elif heure_arrivee<=17:
        plage= "Après-midi"
    else:
        plage= "Soir"
    loi=tableau_duree.loc[plage,lieu]
    dist=loi[0]
    #print(dist)
    param=list(loi[1])
    if abs(param[-2])<10**(-2):
        param[-2]=0
    #print(param)
    sample = getattr(stats,dist).rvs(*param)
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


#c'est un schéma d'algo pour l'instant
def simulation(n=1): #n le nombre d'individu que l'on simule
    
    Jour=[] #nom de la table que l'on va compléter pour tous les individus

    for individu in range(n):
        
        trajet_realise=0 
        lieu_depart = loi_lieu_depart(1)[0]
        temps_attente=duree_lieu(0,lieu_depart)   #Solène et Guilhem parts (intialisation = premier départ)
        temps_trajet=duree_trajet(temps_attente)
        heure_arrivee=temps_attente+temps_trajet #heure d'arrivée premier trajet
        
        while heure_arrivee<24: # (on vérifie à chaque fois qu'on n'a pas fini la journée)
            if trajet_realise == 0: #c'est le premier déplacement
                trajet_realise=1
                lieu_arrivee = calcul_lieu_arrivee(lieu_depart,heure_arrivee)
                Jour.append([individu,lieu_depart,temps_attente, temps_trajet, temps_attente, heure_arrivee, lieu_arrivee,trajet_realise]) 
                #on implémente le lieu de départ, d'arrivée, le temps d'attente et de trajet que l'on a calaculé précedemment
                #on calcule déjà l'heure d'arrivée pour savoir si on a dépassé les 24 heures
                lieu_depart = lieu_arrivee  #lieu arrivee du déplacement précédent
                temps_attente = duree_lieu(heure_arrivee,lieu_depart)    
                heure_depart = heure_arrivee + temps_attente
                temps_trajet= duree_trajet(heure_depart)    #Solène et Guilhem parts
                heure_arrivee =heure_depart+temps_trajet

            else : #à partir du second trajet
                trajet_realise =+ 1
                lieu_arrivee = calcul_lieu_arrivee(lieu_depart,heure_arrivee)
                Jour.append([individu, lieu_depart,temps_attente, temps_trajet,heure_depart, heure_arrivee, lieu_arrivee, trajet_realise])
                lieu_depart= lieu_arrivee  #lieu arrivee du déplacement précédent
                temps_attente= duree_lieu(heure_arrivee,lieu_depart)        #Solène et Guilhem parts
                heure_depart = heure_arrivee+temps_attente #heure de départ du procahin
                temps_trajet=duree_trajet(heure_depart)
                heure_arrivee =+ temps_attente+temps_trajet
   
    
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



"""Script"""
