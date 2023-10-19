import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib import animation
import time
from numba import njit
from tqdm import tqdm
from FenetreParametre import App
from numpy import pi
import matplotlib.cm as cm
import matplotlib.colors as colors
from numba import jit

"""
TO DO LIST:
- LE ROULEMENT
- LE TABLEAU POUR LES VECTEUR NORMAUX ETC...
"""



def trajectoire(POSITION, nb_grains, Agauche, Cgauche, Adroite, Cdroite, paroiGauche, paroiDroite, debut_du_trou, hauteur_bac, largeur_bac_gauche, limite_gauche, limite_droite):
    """
    Affiche la trajectoire des grains dans un graphe matplotlib.

    Paramètres
    ==========
    Retour
    ======
    rien
    """
    print("Affichage de la trajectoire...")

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # dessin du silo dans le tableau graphique matplot
    # on trouve le x du debut du trou pour les deux parois:
    x_debut_du_trou_gauche = (debut_du_trou - Cgauche)/Agauche
    x_debut_du_trou_droite = (debut_du_trou - Cdroite)/Adroite
    X1 = np.linspace(limite_gauche, x_debut_du_trou_gauche, 100)
    X2 = np.linspace(x_debut_du_trou_droite, limite_droite, 100)
    plt.plot(X1, paroiGauche(X1), color='#EEEEEE')
    plt.plot(X2, paroiDroite(X2), color='#EEEEEE')

    # dessin du bac de reception
    X3 = np.linspace(-largeur_bac_gauche, largeur_bac_gauche, 100)
    Y3 = np.zeros(100) + hauteur_bac
    plt.plot(X3, Y3, color='#EEEEEE')
    
    for grain in range(nb_grains):
        ax.plot(POSITION[:, grain, 0], POSITION[:, grain, 1], label="grain {}".format(grain))

    fig.patch.set_facecolor('#222831')                          # On définit la couleur de fond de la figure
    ax.set_facecolor('#222831')                          # On définit la couleur de fond de la figure
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('#EEEEEE')
    ax.grid(alpha=0.1)
    plt.xlim([limite_gauche, limite_droite])
    plt.ylim([limite_bas, limite_haut])
    plt.legend()
    plt.show()


def grain_anime(POSITION, VITESSE, nb_grains, RAYON, Agauche, Cgauche, Adroite, Cdroite, paroiGauche, paroiDroite, debut_du_trou, hauteur_bac, largeur_bac_gauche, largeur_silo_gauche, largeur_silo_droite, nb_temps, pas_de_temps):
    """
    Fait une animation de la chute des grains dans le silo.

    Paramètres
    ==========
    POSITION : np.array, tableau des positions
    hauteur_silo : float, hauteur du silo
    largeur_silo : float, largeur du silo
    nb_grains : int, nombre de grains
    rayon : float, rayon des grains

    Retour
    ======
    rien    
    """
    print("Animation en cours...")

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    
    # dessin du silo dans le tableau graphique matplot
    # on trouve le x du debut du trou pour les deux parois:
    x_debut_du_trou_gauche = (debut_du_trou - Cgauche)/Agauche
    x_debut_du_trou_droite = (debut_du_trou - Cdroite)/Adroite
    X1 = np.linspace(largeur_silo_gauche, x_debut_du_trou_gauche, 100)
    X2 = np.linspace(x_debut_du_trou_droite, largeur_silo_droite, 100)
    plt.plot(X1, paroiGauche(X1), color='#EEEEEE')
    plt.plot(X2, paroiDroite(X2), color='#EEEEEE')
    
    # dessin du bac de reception
    X3 = np.linspace(-largeur_bac_gauche, largeur_bac_gauche, 100)
    Y3 = np.zeros(100) + hauteur_bac
    plt.plot(X3, Y3, color='#EEEEEE')

    # dessin des grains dans le tableau graphique matplot
    couleurs = ['#EEEEEE', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown']
    grains = []
    #texts = []
    for grain in range(nb_grains):
        grains.append(ax.add_patch(patches.Circle((POSITION[0, grain, 0], POSITION[0, grain, 1]), radius=RAYON[grain], fill=True, color=couleurs[grain%len(couleurs)])))
        #texts.append(ax.text(POSITION[0, grain, 0], POSITION[0, grain, 1], str(grain), ha='center', va='center', fontsize=8, color='white'))
    
    time_text = ax.text(0.05, 0.99, '', transform=ax.transAxes, verticalalignment='top', fontsize=12, color='#EEEEEE')
    accelerateur = 25
    def animate(i):
        time_text.set_text(f'Indice temps: {i*accelerateur}/{nb_temps}, temps(s): {i*accelerateur*pas_de_temps:.2f}/{nb_temps*pas_de_temps:.2f}')
        for grain in range(nb_grains):
            vitesse = VITESSE[i*accelerateur, grain]  # Obtention de la vitesse du grain à l'étape temporelle
            couleur = plt.cm.jet(abs(vitesse))  # Calcul de la couleur en fonction de la vitesse
            grains[grain].set_color(couleur[1])  # Mise à jour de la couleur du grain
            grains[grain].center = (POSITION[i*accelerateur, grain, 0], POSITION[i*accelerateur, grain, 1])
            #texts[grain].set_position((POSITION[i*accelerateur, grain, 0], POSITION[i*accelerateur, grain, 1]))
        return grains + [time_text] #+ texts

    
    ani = animation.FuncAnimation(fig, animate, frames=int(POSITION.shape[0]/accelerateur), interval=1, blit=True)
    # Normalisation des valeurs de vitesse
    norm = colors.Normalize(vmin=np.min(abs(VITESSE)), vmax=np.max(abs(VITESSE)))
    # Création de l'échelle de couleur
    cmap = cm.ScalarMappable(norm=norm, cmap='jet')
    cb = plt.colorbar(cmap)
    cb.set_label('Vitesse', color='#EEEEEE')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="#EEEEEE")
    cb.ax.yaxis.set_tick_params(color='#EEEEEE')
    fig.patch.set_facecolor('#222831')                          # On définit la couleur de fond de la figure
    ax.set_facecolor('#222831')                          # On définit la couleur de fond de la figure
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('#EEEEEE')
    ax.grid(alpha=0.1)
    plt.xlim([limite_gauche, limite_droite])
    plt.ylim([limite_bas, limite_haut])
    plt.show()








    #-------------------------------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------FONCTIONS-----------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------------------------------#




@njit
def application_efforts_distance(masse):
    """
    Application des efforts à distance (par exemple la pesanteur).

    Paramètres
    ==========
    masse : float, masse du grain

    Retour
    ======
    forces : np.array, forces appliquée au grain
    """
    return np.array([0, -masse*9.81])

@njit
def allongement_normal_grain_grain(position_i, position_j, rayon_i, rayon_j):
    """
    Calcul de l'allongement normal entre deux grains i et j à partir de l'équation

    Paramètres
    ==========
    position_i : np.array, position du grain i
    position_j : np.array, position du grain j

    Retour
    ======
    allongement_normal : float, allongement normal entre les grains i et j
    """

    return np.sqrt((position_i[0] - position_j[0])**2 + (position_i[1] - position_j[1])**2) - (rayon_i + rayon_j)


# Distance à la paroi, droite d'équation: A*x + B*y + C = 0. Ici B=1, A=-Agauche/droite et C=-Cgauche/droite.
# La distance orthogonale est alors donné par la relation : d = abs(A*x + B*y + C) / sqrt(A**2 + B**2)
@njit
def allongement_normal_grain_paroigauche(position, rayon, Agauche, Cgauche):
    """
    Calcul de l'allongement normal entre un grain et la paroi gauche à partir de l'équation

    Paramètres
    ==========
    position : np.array, position du grain
    rayon : float, rayon du grain
    Agauche : float, coefficient directeur de la paroi gauche
    Cgauche : float, ordonnée à l'origine de la paroi gauche

    Retour
    ======
    penetration_gauche : float, allongement normal entre le grain et la paroi gauche
    """

    distance_a_la_gauche = abs(-Agauche * position[0] + 1*position[1] - Cgauche) / np.sqrt(Agauche**2 + 1)
    penetration_gauche = distance_a_la_gauche - rayon

    return penetration_gauche

@njit
def allongement_normal_grain_paroidroite(position, rayon, Adroite, Cdroite):
    """
    Calcul de l'allongement normal entre un grain et la paroi droite à partir de l'équation

    Paramètres
    ==========
    position : np.array, position du grain
    rayon : float, rayon du grain
    Adroite : float, coefficient directeur de la paroi droite
    Cdroite : float, ordonnée à l'origine de la paroi droite

    Retour
    ======
    penetration_droite : float, allongement normal entre le grain et la paroi droite
    """
    distance_a_la_droite = abs(-Adroite * position[0] + 1*position[1] - Cdroite) / np.sqrt(Adroite**2 + 1)
    penetration_droite = distance_a_la_droite - rayon

    return penetration_droite

@njit
def allongement_tangentiel_grain_paroi_gauche(vitesse_i, vecteur_tangent_paroi_gauche, pas_de_temps, allongement_tangentiel):
    """
    """

    produit_scalaire = np.dot(vitesse_i , vecteur_tangent_paroi_gauche)

    return allongement_tangentiel + produit_scalaire*pas_de_temps

@njit
def allongement_tangentiel_grain_paroi_droite(vitesse_i, vecteur_tangent_paroi_droite, pas_de_temps, allongement_tangentiel):
    """
    """
    produit_scalaire = np.dot(vitesse_i , vecteur_tangent_paroi_droite)

    return allongement_tangentiel + produit_scalaire*pas_de_temps

@njit
def allongement_tangentiel_grain_grain(position_i, position_j, vitesse_i, vitesse_j, pas_de_temps, allongement_tangentiel):
    """
    Calcul de l'allongement tangentiel entre deux grains i et j à partir de l'équation

    Paramètres
    ==========
    POSITION : np.array, position des grains
    VITESSE : np.array, vitesse des grains
    i : int, indice du grain i
    j : int, indice du grain j
    indice_temps : int, indice du temps
    
    Retour
    ======
    allongement_tangentiel : float, allongement tangentiel entre les grains i et j
    """
    vecteur_normal = (position_i - position_j)/(np.linalg.norm(position_i - position_j))
    vecteur_tangent = np.array([-vecteur_normal[1], vecteur_normal[0]])
    produit_scalaire = np.dot(vitesse_i - vitesse_j, vecteur_tangent)

    return allongement_tangentiel + produit_scalaire*pas_de_temps

@njit
def derivee_allongement_normal_grain_grain(vitesse_i, vitesse_j, vecteur_normal):
    """
    Calcul de la dérivée de l'allongement/distance normal à partir de l'équation
    Paramètres
    ==========

    Retour
    ======
    """
    vitesse_relative = vitesse_i - vitesse_j
    derivee_allongement = np.dot(vitesse_relative, vecteur_normal)
        
    return derivee_allongement

@njit
def actualisation_1(mise_a_jour, POSITION, VITESSE_DEMI_PAS, VITESSE, ACCELERATION, GRILLE, indice_temps, pas_de_temps, nb_grains, c, limite_gauche):
    """
    Fonction qui actualise la grille, la position et la vitesse des grains à l'instant k

    Paramètres
    ==========


    Retour
    ======
    GRILLE : np.array, grille contenant les grains
    POSITION : np.array, position des grains
    VITESSE : np.array, vitesse des grains
    """
    # Actualisation position et vitesse
    POSITION[indice_temps] = POSITION[indice_temps-1] + VITESSE_DEMI_PAS[indice_temps-1]*pas_de_temps
    VITESSE[indice_temps] = VITESSE_DEMI_PAS[indice_temps-1] + ACCELERATION[indice_temps-1]*pas_de_temps

    # Grille
    for grain, maj in enumerate(mise_a_jour):
        if maj:
            # On associe à chaque case de la grille les grains qui sont dans cette case
            # Probleme car pos_case peut etre negatif pour ca on déplace le repere:
            try:
                pos_case = (int((POSITION[indice_temps, grain, 0] + abs(limite_gauche))/c), int((POSITION[indice_temps,grain,1]+ abs(limite_gauche))/c))
                GRILLE[pos_case[0], pos_case[1], grain] = 1
            except:
                #on stope les grains qui sont dans le bac:
                VITESSE[indice_temps, grain] = np.array([0,0])
                ACCELERATION[indice_temps, grain] = np.array([0,0])
                VITESSE_DEMI_PAS[indice_temps, grain] = np.array([0,0])
                # et on arrete de les mettres a jour:
                # Trouver l'index de tous les éléments qui ne sont pas égaux à 'grain1'
                mise_a_jour[grain] = 0


    return GRILLE, POSITION, VITESSE, mise_a_jour

@njit
def voisinage(mise_a_jour, grain, x, y, GRILLE):
    """
    Détermine la liste de voisinage du grain i

    Paramètres
    ==========
    grain : int, indice du grain
    x : int, indice de la ligne du grain
    y : int, indice de la colonne du grain
    GRILLE : np.array, grille des grains

    Retour
    ======
    voisinage : liste, tableau des indices des grains en contact avec le grain i
    """
    voisinage = []

    for j in range(-1, 2):
        for k in range(-1, 2):
            for i, bit in enumerate(GRILLE[x+j,y+k]):
                if bit and mise_a_jour[i] and i != grain:
                    voisinage.append(i)


    return voisinage

@njit
def maj_contact(GRILLE, mise_a_jour, indice_temps, nb_grains, POSITION, RAYON, Agauche, Adroite, Cgauche, Cdroite, limite_gauche, ALLONGEMENT, VITESSE, debut_du_trou):
    """
    Met à jour la liste des contacts

    Paramètres
    ==========

    Retour
    ======
    """

    nouveau_contact = np.zeros((nb_grains, nb_grains+2, 1), dtype=np.int64)
    nouveau_allongement = np.zeros((nb_grains, nb_grains+2, 2), dtype=np.float64)
    for i, maj in enumerate(mise_a_jour):
        if maj:
            pos_i = POSITION[indice_temps, i]
            vitesse_i = VITESSE[indice_temps, i]
            rayon_i = RAYON[i]

            # Contact avec les parois ?
            if pos_i[1] >= debut_du_trou:
                penetration_gauche = allongement_normal_grain_paroigauche(pos_i, rayon_i, Agauche, Cgauche)
                penetration_droite = allongement_normal_grain_paroidroite(pos_i, rayon_i, Adroite, Cdroite)

                if penetration_gauche < 0:
                    nouveau_contact[i, nb_grains] = 1
                    nouveau_allongement[i, nb_grains, 0] = penetration_gauche
                    allongement_tangentiel = ALLONGEMENT[i, nb_grains, 1]
                    allongement_tangentiel = allongement_tangentiel_grain_paroi_droite(vitesse_i, vecteur_tangent_paroi_droite, pas_de_temps, allongement_tangentiel)
                    nouveau_allongement[i, nb_grains, 1] = allongement_tangentiel

                elif penetration_droite < 0:
                    nouveau_contact[i, nb_grains+1] = 1
                    nouveau_allongement[i, nb_grains+1, 0] = penetration_droite
                    allongement_tangentiel = ALLONGEMENT[i, nb_grains+1, 1]
                    allongement_tangentiel = allongement_tangentiel_grain_paroi_gauche(vitesse_i, vecteur_tangent_paroi_gauche, pas_de_temps, allongement_tangentiel)
                    nouveau_allongement[i, nb_grains+1, 1] = allongement_tangentiel

            # Contact avec un autre grain ?
            pos_case = (int((pos_i[0] + abs(limite_gauche))/c), int((pos_i[1] + abs(limite_gauche))/c))
            voisins = voisinage(mise_a_jour, i, pos_case[0], pos_case[1], GRILLE)
            for j in voisins:
                if i != j: # pas besoin de check le mise a jour car 'voisins' ne peut pas contenir les grains pas à jour
                    pos_j = POSITION[indice_temps, j]
                    vitesse_j = VITESSE[indice_temps, j]
                    rayon_j = RAYON[j]

                    allongement_normal = allongement_normal_grain_grain(pos_i, pos_j, rayon_i, rayon_j)
                    if allongement_normal < 0:
                        nouveau_contact[i, j] = 1
                        nouveau_allongement[i, j, 0] = allongement_normal
                        allongement_tangentiel = ALLONGEMENT[i, j, 1]
                        allongement_tangentiel = allongement_tangentiel_grain_grain(pos_i, pos_j, vitesse_i, vitesse_j, pas_de_temps, allongement_tangentiel)
                        nouveau_allongement[i, j, 1] = allongement_tangentiel


    print(nouveau_allongement)        
    return nouveau_contact, nouveau_allongement


@njit
def resultante_et_actualisation_2(coefficient_frottement, CONTACT, POSITION, VITESSE, ACCELERATION, VITESSE_DEMI_PAS, MASSE, RAYON, AMORTISSEMENT, raideur_normale, raideur_tangentielle, coefficient_trainee, nb_grains, indice_temps, pas_de_temps, c, Agauche, Cgauche, Adroite, Cdroite, limite_gauche, debut_du_trou, mise_a_jour, hauteur_bac, vecteur_orthogonal_paroi_gauche, vecteur_orthogonal_paroi_droite, vecteur_tangent_paroi_gauche, vecteur_tangent_paroi_droite):
    """
    Fonction qui calcule la force résultante et actualise l'accélération à l'instant k et la vitesse des grains à l'instant k+1/2

    Paramètres
    ==========


    Retour
    ======
    """

    for grain1, maj in enumerate(mise_a_jour):

        if maj:
            #Variables utiles:
            position_grain1 = POSITION[indice_temps, grain1]
            vitesse_grain1 = VITESSE[indice_temps, grain1]
            masse_grain1 = MASSE[grain1]
            rayon_grain1 = RAYON[grain1]
            tableau_contact_grain1 = CONTACT[grain1] # de la forme (nbgrains+2, 1)


            #Initialisation force résultante:
            force_resultante = application_efforts_distance(masse_grain1) #Force à distance = gravité

            #Initialisation force de contact:
            force_contact = np.array([0.0, 0.0])

            for contact, en_contact in enumerate(tableau_contact_grain1):

                # S'il y a contact:
                if en_contact:

                    # Rencontre avec une paroi du silo ?
                    # Paroi gauche
                    if contact == nb_grains:
                        penetration_gauche = ALLONGEMENT[grain1, nb_grains, 0]
                        # Effort normal:
                        force_normale = -raideur_normale * penetration_gauche * vecteur_orthogonal_paroi_gauche
                        force_contact += force_normale
                        print("----------------------")
                        print("penetration gauche:", penetration_gauche)
                        print("----------------------")
                            
                        """
                        #Effort tangentiel:
                        allongement_tangentiel = ALLONGEMENT[grain1, nb_grains][1]
                        if np.dot(vecteur_tangent_paroi_gauche, VITESSE[indice_temps,grain1]) < 0:
                            vecteur_tangent_paroi_gauche = -vecteur_tangent_paroi_gauche
                        force_contact += -raideur_tangentielle * allongement_tangentiel * vecteur_tangent_paroi_gauche
                        if np.linalg.norm(force_contact) > 100 or np.linalg.norm(vitesse_grain1)>2:
                            print("----------------------")
                            print("grain gauche:", grain1)
                            print("liste contact grain gauche:", tableau_contact_grain1)
                            print("allongement tangentiel gauche:", allongement_tangentiel)
                            print("vecteur tangentiel gauche:", vecteur_tangent_paroi_gauche)
                            print("force contact gauche:", force_contact)
                            print("raideur tangentielle gauche:", raideur_tangentielle)
                            print("norme force contact tangentielle gauche:", np.linalg.norm(force_contact))
                            print("vitesse grain gauche:", VITESSE[indice_temps,grain1])
                            print("----------------------")
                        """
                        """
                        # Glissement
                        if abs(-raideur_tangentielle * allongement_tangentiel) <= coefficient_frottement * np.linalg.norm(force_normale):
                            force_tangentielle = -raideur_tangentielle * allongement_tangentiel * vecteur_tangent_paroi_gauche
                            force_contact += force_tangentielle
                        else:
                            force_tangentielle = np.sign(allongement_tangentiel * -raideur_tangentielle) * coefficient_frottement * np.linalg.norm(force_normale) * vecteur_tangent_paroi_gauche
                            force_contact += force_tangentielle
                        """


                    # Paroi droite
                    elif contact == nb_grains+1:
                        penetration_droite = ALLONGEMENT[grain1, nb_grains+1, 0]
                        print("----------------------")
                        print("penetration droite:", penetration_droite)
                        print("----------------------")
                        # Effort normal:
                        force_normale = -raideur_normale * penetration_droite * vecteur_orthogonal_paroi_droite
                        force_contact += force_normale
                        """
                        #Effort tangentiel:
                        allongement_tangentiel = allongement_tangentiel_grain_paroi_droite(vecteur_tangent_paroi_droite, indice_temps, VITESSE, grain, pas_de_temps, indice_impact)
                        if np.dot(vecteur_tangent_paroi_droite, VITESSE[indice_temps,grain1]) < 0:
                            vecteur_tangent_paroi_droite = -vecteur_tangent_paroi_droite
                        force_contact += -raideur_tangentielle * allongement_tangentiel * vecteur_tangent_paroi_droite
                        """
                        """
                        # Glissement
                        if abs(-raideur_tangentielle * allongement_tangentiel) <= coefficient_frottement * np.linalg.norm(force_normale):
                            force_tangentielle = -raideur_tangentielle * allongement_tangentiel * vecteur_tangent_paroi_droite
                            force_contact += force_tangentielle
                        else:
                            force_tangentielle = np.sign(allongement_tangentiel * -raideur_tangentielle) * coefficient_frottement * np.linalg.norm(force_normale) * vecteur_tangent_paroi_droite
                            force_contact += force_tangentielle
                        
                        """
                    

                    # Rencontre avec un autre grain ?
                    else:
                        # Variables utiles:
                        grain2 = contact
                        position_grain2 = POSITION[indice_temps, grain2]

                        # Effort normal:
                        vecteur_normal_inter_grain = (position_grain1 - position_grain2)/(np.linalg.norm(position_grain1 - position_grain2))
                        allongement_normal = ALLONGEMENT[grain1, grain2, 0]
                        force_normale = -raideur_normale * allongement_normal * vecteur_normal_inter_grain
                        force_contact += force_normale
                        print("----------------------")
                        print("Allongement normal:", allongement_normal)
                        print("----------------------")
                        
                        
                        
                        # Effort tangentiel:
                        vecteur_tangentiel_inter_grain = np.array([-vecteur_normal_inter_grain[1], vecteur_normal_inter_grain[0]])
                        allongement_tangentiel = ALLONGEMENT[grain1, grain2, 1]
                        force_contact += -raideur_tangentielle * allongement_tangentiel * vecteur_tangentiel_inter_grain
                        if np.linalg.norm(force_contact) > 1:
                            print("----------------------")
                            print("indice temps:", indice_temps)
                            print("grain1:", grain1)
                            print("grain2:", grain2)
                            print("allongement tangentiel:", allongement_tangentiel)
                            print("vecteur tangentiel:", vecteur_tangentiel_inter_grain)
                            print("force contact:", force_contact)
                            print("vitesses:", VITESSE[indice_temps,grain1], VITESSE[indice_temps,grain2])
                            print("----------------------")
                        
                       
                        """
                        # Amortissement:
                        derivee_allongement_normal = derivee_allongement_normal_grain_grain(vitesse_grain1, vitesse_grain2, vecteur_normal_inter_grain)
                        force_contact += - AMORTISSEMENT[grain1] * derivee_allongement_normal * vecteur_normal_inter_grain
                        
                        # Glissement
                        if abs(-raideur_tangentielle * allongement_tangentiel) <= coefficient_frottement * np.linalg.norm(force_normale):
                            force_tangentielle = -raideur_tangentielle * allongement_tangentiel * vecteur_tangentiel_inter_grain 
                            force_contact += force_tangentielle
                        else:
                            force_tangentielle = np.sign(allongement_tangentiel * -raideur_tangentielle) * coefficient_frottement * np.linalg.norm(force_normale) * vecteur_tangentiel_inter_grain
                            force_contact += force_tangentielle
                        """

            # Mise à jour de la résultante des forces sur grain1
            force_resultante += force_contact




            # Rencontre avec le bac du silo ?
            distance_bac = position_grain1[1] - hauteur_bac
            penetration_bac = distance_bac - rayon_grain1
            if penetration_bac < 0: 
                #on stope les grains qui sont dans le bac:
                VITESSE[indice_temps, grain1] = np.array([0,0])
                ACCELERATION[indice_temps, grain1] = np.array([0,0])
                VITESSE_DEMI_PAS[indice_temps, grain1] = np.array([0,0])
                # et on arrete de les mettre a jour:
                mise_a_jour[grain1] = 0
            
            # Le reste:
            # Force de trainée:
            norme_vitesse = np.linalg.norm(vitesse_grain1)
            if norme_vitesse > 0:
                frotemment_air = (1/2)*rho*(4*pi*rayon_grain1**2)*coefficient_trainee*norme_vitesse**2
                vecteur_directeur_vitesse = vitesse_grain1/ norme_vitesse
                force_resultante += -frotemment_air*vecteur_directeur_vitesse
            # Calcul de l'accélération du grain à partir de l'équation
            ACCELERATION[indice_temps, grain1] = force_resultante / masse_grain1
            # Calcul de la vitesse de demi-pas à k+1/2 à partir de l'équation
            VITESSE_DEMI_PAS[indice_temps, grain1] = VITESSE_DEMI_PAS[indice_temps-1, grain1] + ACCELERATION[indice_temps, grain1] * pas_de_temps / 2



    return mise_a_jour, ACCELERATION, VITESSE_DEMI_PAS





#-------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------DEFINITION----------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------#





# Remarque:
# - La raideur doit être tres grande si la masse volumique est importante
#   ce qui implique qu'on doit baisser en csqe le pas de temps


# Infos:
# masse = rho * volume, volume = 4/3 * pi * rayon^3, masse/rayon = rho * 4/3 * pi * rayon^2
# force_trainée = 1/2 * rho * surface * coefficient_trainee * vitesse^2

# Matériaux:
#  Sable(gros): 1e-3 m, rho = 2700 kg/m3:
#   - raideur_normale = rho*5 #N/m
#   - pas_de_temps = (1/raideur_normale)*5 #s
# Gravier grossier : rayon de 10 à 30 mm, Gravier calcaire : 1600 à 2000 kg/m³:
"""
Pour un grain de blé circulaire, voici quelques valeurs typiques :

Diamètre moyen d'un grain de blé : 5 à 7 millimètres (0,005 à 0,007 mètres)
Masse volumique du blé : environ 780 à 820 kg/m³
Coefficient de traînée (Cd) pour un grain de blé : environ 0,47
"""
# ON PLACE LE REPERE EN BAS A GAUCHE (0,0) DU SILO COIN BAS GAUCHE Y VERS LE HAUT X VERS LA DROITE.

# Définition grain
coef = 3
nb_grains = 60
rayon = 5e-3 #m
rho = 770 #kg/m3
RAYON = np.random.uniform(low=rayon*0.8, high=rayon*1.2, size=nb_grains)
MASSE = rho * 4/3 * pi * RAYON**3
raideur_normale = rho #N/m
raideur_tangentielle = (1/2)*raideur_normale #N/m
coefficient_trainee = 0.10
AMORTISSEMENT = np.sqrt(raideur_normale*MASSE)*0.1

# ESPACE
limite_bas = -1  #m
limite_haut = 2.5 #m
limite_gauche = -1 #m
limite_droite = 1 #m
coefficient_frottement = 0.5
# Definition bac de réception
hauteur_bac = 0.4 #m
largeur_bac_gauche = 0.5/2 #m
largeur_bac_droite = 0.5/2 #m
# Définition de la grille
c = 2*rayon*1.2 #pas d'espace de la grille en m
# On définit une grille pour discrétiser l'espace selon le pas d'espace c, a chaque case on met la liste des grains qui sont dans cette case
nb_cases_x = int((limite_droite - limite_gauche)/c) + 2
nb_cases_y = int((limite_haut - limite_bas)/c) + 2
GRILLE = np.zeros(( nb_cases_x , nb_cases_y, nb_grains), dtype=int) #on définit une grille vide #ancienne version : GRILLE = {(i,j):[] for i in range(int(limite_gauche/c)-1, int(limite_droite/c)+2) for j in range(int(limite_bas/c)-1, int(limite_haut/c)+2)}


# TEMPS
temps = 0
indice_temps = 0
pas_de_temps = np.sqrt(np.mean(MASSE)/raideur_normale)*0.2#s
duree_simulation = 2 #s
nb_temps = int(duree_simulation/pas_de_temps)


#TABLEAUX DE DONNEES:
POSITION = np.zeros((nb_temps, nb_grains, 2))   
VITESSE = np.zeros((nb_temps, nb_grains, 2))
VITESSE_DEMI_PAS = np.zeros((nb_temps, nb_grains, 2))
ACCELERATION = np.zeros((nb_temps, nb_grains, 2))
CONTACT = np.zeros((nb_grains, nb_grains+2, 1), dtype=np.int64)
ALLONGEMENT = np.zeros((nb_grains, nb_grains+2, 2), dtype=np.float64)

mise_a_jour = np.array([1 for i in range(nb_grains)])  #liste qui permet de savoir si on doit mettre à jour le grain ou pas

# Affichage des infos implicites:
print(f"nombre de grain: {nb_grains}")
print(f"pas de temps: {pas_de_temps:.2E} s.")
print(f"nombre de temps: {nb_temps}.")
print(f"raideur normale: {raideur_normale:.2E} N/m.")
print(f"masse moyenne des grains: {np.mean(MASSE):.2E} kg.")
print(f"rayon moyen des grains: {np.mean(RAYON):.2E} m.")
print(f"amoortissement moyen: {np.mean(AMORTISSEMENT):.2E} Ns/m.")

    #-------------------------------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------SIMULATION----------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------------------------------#



if __name__ == "__main__":


    app = App()
    app.racine.mainloop()
    plt.close('all')

    # On définit les droites des parois des silos comme des droites de la forme y = Ax + C afin de mettre sous la forme -Ax + y - C = 0

    CoeffDir = app.CoeffDir
    OrdOrigine = app.OrdOrigine
    debut_du_trou = app.debutTrou
    Agauche = CoeffDir
    Adroite = -CoeffDir
    Cgauche = OrdOrigine
    Cdroite = OrdOrigine

    paroiGauche = lambda x : Agauche*x + Cgauche
    vecteur_directeur_paroi_gauche = np.array([1.0, Agauche])/ np.sqrt(1 + (Agauche)**2) #pointe vers le haut, normalisé
    vecteur_orthogonal_paroi_gauche = np.array([-Agauche, 1.0])/ np.sqrt(1 + (Agauche)**2) #pointe vers l'intérieur du silo, normalisé
    paroiDroite = lambda x : Adroite*x+Cdroite
    vecteur_directeur_paroi_droite = np.array([1.0, Adroite])/ np.sqrt(1 + (Agauche)**2) #pointe vers le haut, normalisé
    vecteur_orthogonal_paroi_droite = np.array([-Adroite, 1.0])/ np.sqrt(1 + (Agauche)**2) #pointe vers l'intérieur du silo, normalisé
    vecteur_tangent_paroi_gauche = np.array([-vecteur_orthogonal_paroi_gauche[1], vecteur_orthogonal_paroi_gauche[0]])
    vecteur_tangent_paroi_droite = np.array([-vecteur_orthogonal_paroi_droite[1], vecteur_orthogonal_paroi_droite[0]])
    #Positionnement initiale des grains
    hauteur = app.hauteur #m
    gauche = (hauteur - Cgauche)/Agauche + rayon*1.3
    droite = (hauteur - Cdroite)/Adroite - rayon*1.3
    grain = 0 # compteur grain
    q = 0 # compteur colonne
    while grain < nb_grains:
        while True:
            x = gauche + (rayon*1.3*2)*q
            y = hauteur
            if x > droite or grain >= nb_grains:
                break
            else:
                POSITION[0, grain, 0] = x
                POSITION[0, grain, 1] = y
                grain += 1
                q += 1
        if grain < nb_grains:
            q = 0
            hauteur -= rayon*1.3*2
            gauche = (hauteur - Cgauche)/Agauche + rayon*1.3
            droite = (hauteur - Cdroite)/Adroite - rayon*1.3
            x = gauche + (rayon*1.3*2)*q
            y = hauteur
            POSITION[0, grain, 0] = x
            POSITION[0, grain, 1] = y
            grain += 1
            q += 1
        else:
            break














    # Boucle principale
    print("Simulation en cours...")
    start_time = time.time()
    for indice_temps in tqdm(range(1, nb_temps)):
        # Actualisation du temps
        temps += pas_de_temps
        
        # Actualisation de la grille, de la position et de la vitesse
        GRILLE, POSITION, VITESSE, mise_a_jour = actualisation_1(mise_a_jour,POSITION, VITESSE_DEMI_PAS, VITESSE, ACCELERATION, GRILLE, indice_temps, pas_de_temps, nb_grains, c, limite_gauche)   

        #On met à jour la liste des contacts:
        CONTACT, ALLONGEMENT = maj_contact(GRILLE, mise_a_jour, indice_temps, nb_grains, POSITION, RAYON, Agauche, Adroite, Cgauche, Cdroite, limite_gauche, ALLONGEMENT, VITESSE, debut_du_trou)

        # Calcul des efforts de contact pour mise à jour des vitesses à temps k+1/2 et accélérations à temps k
        mise_a_jour, ACCELERATION, VITESSE_DEMI_PAS = resultante_et_actualisation_2(coefficient_frottement, CONTACT, POSITION, VITESSE, ACCELERATION, VITESSE_DEMI_PAS, MASSE, RAYON, AMORTISSEMENT, raideur_normale, raideur_tangentielle, coefficient_trainee, nb_grains, indice_temps, pas_de_temps, c, Agauche, Cgauche, Adroite, Cdroite, limite_gauche, debut_du_trou, mise_a_jour, hauteur_bac, vecteur_orthogonal_paroi_gauche, vecteur_orthogonal_paroi_droite, vecteur_tangent_paroi_gauche, vecteur_tangent_paroi_droite)
        
        # Pour éviter les doublons dans la prochaine case de la grille on la réinitialise:
        GRILLE = np.zeros(( nb_cases_x , nb_cases_y, nb_grains), dtype=int)

    # Fin de la boucle principale
    print("Fin de la simulation")
    print("Temps de calcul: ", time.time() - start_time, "secondes")

    #Affichage:
    trajectoire(POSITION, nb_grains, Agauche, Cgauche, Adroite, Cdroite, paroiGauche, paroiDroite, debut_du_trou, hauteur_bac, largeur_bac_gauche, limite_gauche, limite_droite)
    grain_anime(POSITION, VITESSE, nb_grains, RAYON, Agauche, Cgauche, Adroite, Cdroite, paroiGauche, paroiDroite, debut_du_trou, hauteur_bac, largeur_bac_gauche, limite_gauche, limite_droite, nb_temps, pas_de_temps)
