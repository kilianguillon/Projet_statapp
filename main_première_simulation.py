#Librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import scipy.stats as stats
import ast
from scipy.stats import invgamma,loggamma,invgauss

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
    sample=0
    if loi[0] == 'norm':
        # Paramètres de la distribution normale (moyenne, écart-type)
        # Générer un seul nombre suivant la distribution normale
        sample = np.random.normal(loc=round(loi[1][0],4), scale=round(loi[1][1],4))
    elif loi[0]  == 'beta':
        # Paramètres de la distribution bêta (a, b)
        # Générer un seul nombre suivant la distribution bêta
        sample = np.random.beta(a=round(loi[1][0],4), b=round(loi[1][-1],4))
    elif loi[0]  == 'gamma':
        # Paramètres de la distribution gamma (a, scale)
        # Générer un seul nombre suivant la distribution gamma
        sample = np.random.gamma(a=round(loi[1][0],4),loc=round(loi[1][1]), scale=round(loi[1][2],4))
    elif loi[0] == 'pareto':
        # Paramètres de la distribution de Pareto (b, loc, scale)
        sample = np.random.pareto(a=loi[1][0])
    elif loi[0] == 't':           
        # Paramètres de la distribution t de Student (df, loc, scale)
        sample = np.random.standard_t(df=loi[1][0], loc=loi[1][1], scale=loi[1][2])
    elif loi[0] == 'lognorm':
        # Paramètres de la distribution log-normale (s, loc, scale)
        sample = np.random.lognormal(mean=loi[1][1], sigma=loi[1][0])
    elif loi[0] == 'invgamma':
        # Paramètres de la distribution inverse-gamma (a, loc, scale)
         sample = invgamma.rvs(a=loi[1][0], scale=loi[1][-1])
    elif loi[0] == 'loggamma':
        # Paramètres de la distribution log-gamma (c, loc, scale)
        sample = loggamma.rvs(c=loi[1][0], scale=loi[1][-1])
    elif loi[0] == 'invgauss':
         # Paramètres de la distribution inverse-Gaussienne (mu, loc, scale)
        sample = invgauss.rvs(mu=loi[1][0], loc=loi[1][1], scale=loi[1][2])
    elif loi[0] == 'chi2':
        # Paramètres de la distribution de chi carré (df, loc, scale)
        sample = np.random.chisquare(df=loi[1][0])
    if sample<0:
        sample=-sample
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
                Jour.append([individu, lieu_depart,temps_attente, temps_trajet,heure_depart, heure_arrivee, calcul_lieu_arrivee(lieu_depart,heure_arrivee), trajet_realise])
                lieu_depart= lieu_arrivee  #lieu arrivee du déplacement précédent
                temps_attente= duree_lieu(heure_arrivee,lieu_depart)        #Solène et Guilhem parts
                heure_depart = heure_arrivee+temps_attente #heure de départ du procahin
                temps_trajet=duree_trajet(heure_depart)
                heure_arrivee =+ temps_attente+temps_trajet
   
    
    Jour = pd.DataFrame(Jour, columns=["Individu","Lieu_depart","Temps_attente","Temps_trajet","Heure_depart","Heure_arrivee","Lieu_arrivee","Numero_trajet"])
    return Jour.sort_values(by='Heure_arrivee', ascending=False)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

def plot_individual_travels_final(travel_data):
    """
    Final version to plot the daily travels of an individual including parking information before the first departure
    and after the last arrival until midnight. This version ensures uniform line thickness and better visibility
    for parking information.
    
    Parameters:
    - travel_data: DataFrame containing the travel information for an individual, including departure and arrival times,
      parking locations, departure locations, trip number, and parking information.
    """
    # Ensure the data is sorted by departure time
    travel_data_sorted = travel_data.sort_values(by='HEURE_DEPART')
    
    # Start plotting
    fig, ax = plt.subplots(figsize=(12, 2))
    
    # Initial and final parking locations, and their abbreviations
    initial_parking_location = travel_data_sorted.iloc[0]['Lieu_Depart']
    final_parking_location = travel_data_sorted.iloc[-1]['Lieu_Arrivee']
    parking_abbreviations = {'Emplacement privé, gratuit (y compris domicile)': 'EPG',
                             'Emplacement dans un parking public ou commercial payant': 'EPP',
                             'Sur la voie publique': 'VP'}
    
    # Color map for parking locations
    color_map = {'Initial Parking': 'lightgray', 'Final Parking': 'darkgray'}
    colors = plt.cm.tab10.colors  # Using tab10 colormap for up to 10 parking locations
    
    # Plot initial parking segment (from midnight to first departure)
    first_departure = travel_data_sorted.iloc[0]['HEURE_DEPART']
    ax.plot([0, first_departure], [1, 1], color='lightgray', linewidth=8)
    
    # Loop through each trip to plot
    for index, row in travel_data_sorted.iterrows():
        start = row['HEURE_DEPART']
        end = row['HEURE_ARRIVEE']
        parking_location = row['Lieu_Arrivee']
        parking_info = row.get('Stationnement', 'Unknown')
        
        # Assign a color for each parking location (if not initial or final parking)
        if parking_location not in color_map:
            color_map[parking_location] = colors[len(color_map) % len(colors)]
        
        # Plot the travel segment
        ax.plot([start, end], [1, 1], color='black', linewidth=8)  # Uniform line thickness for travel
        
        # Plot the parking segment with slight spacing
        if index < len(travel_data_sorted) - 1:
            next_start = travel_data_sorted.iloc[index + 1]['HEURE_DEPART']
            ax.plot([end, next_start], [1, 1], color=color_map[parking_location], linewidth=8)  # Uniform line thickness for parking
            # Add abbreviated text for parking information, placed closer to the line
            abbreviation = parking_abbreviations.get(parking_info, 'Unk')
            ax.text((end + next_start) / 2, 1.02, abbreviation, ha='center', va='bottom', fontsize=8, backgroundcolor='white')
    
    # Plot final parking segment (from last arrival to midnight)
    last_arrival = travel_data_sorted.iloc[-1]['HEURE_ARRIVEE']
    ax.plot([last_arrival, 24], [1, 1], color='darkgray', linewidth=8)  # Uniform line thickness for final parking
    
    # Improving the plot aesthetics
    ax.set_xlim(0, 24)  # Set x-axis to span from midnight to midnight
    ax.set_yticks([])  # Hide y-axis as it's not relevant
    ax.set_xlabel("Heure")
    plt.title("Déplacements journaliers d'un individu avec stationnements initial et final")
    # Create legend entries for parking locations
    legend_entries = [mpatches.Patch(color=color, label=label) for label, color in color_map.items() if label not in ['Initial Parking', 'Final Parking']]
    # Optionally, add entries for initial and final parking if desired
    legend_entries.append(mpatches.Patch(color='lightgray', label='Parking initial'))
    legend_entries.append(mpatches.Patch(color='darkgray', label='Parking final'))
    # Add legend to the plot
    ax.legend(handles=legend_entries, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage with the provided mock DataFrame
# Ensure your DataFrame includes 'Stationnement' column with parking information.


"""Script"""
