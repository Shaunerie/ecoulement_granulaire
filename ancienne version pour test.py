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


"""
L'algorithme utilisé provient du pdf du professeur 'Mecanique_des_Materiaux_Granulaires_16_17.pdf'

Hypothèses persos:
- Les grains sont des sphères de même masse et de même rayon
- On est dans le vide
"""

def trajectoire(POSITION, nb_grains, Agauche, Cgauche, Adroite, Cdroite, paroiGauche, paroiDroite, debut_du_trou, hauteur_bac, largeur_bac_gauche, largeur_bac_droite, largeur_silo_gauche, largeur_silo_droite):
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
    X1 = np.linspace(largeur_silo_gauche, x_debut_du_trou_gauche, 100)
    X2 = np.linspace(x_debut_du_trou_droite, largeur_silo_droite, 100)
    plt.plot(X1, paroiGauche(X1), color='black')
    plt.plot(X2, paroiDroite(X2), color='black')

    # dessin du bac de reception
    X3 = np.linspace(-largeur_bac_gauche, largeur_bac_gauche, 100)
    Y3 = np.zeros(100) + hauteur_bac
    plt.plot(X3, Y3, color='black')
    
    for grain in range(nb_grains):
        ax.plot(POSITION[:, grain, 0], POSITION[:, grain, 1], label="grain {}".format(grain))

    plt.grid()
    plt.legend()
    plt.show()

def grain_anime(POSITION, nb_grains, rayon, Agauche, Cgauche, Adroite, Cdroite, paroiGauche, paroiDroite, debut_du_trou, hauteur_bac, largeur_bac_gauche, largeur_bac_droite, largeur_silo_gauche, largeur_silo_droite):
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
    plt.plot(X1, paroiGauche(X1), color='black')
    plt.plot(X2, paroiDroite(X2), color='black')
    
    # dessin du bac de reception
    X3 = np.linspace(-largeur_bac_gauche, largeur_bac_gauche, 100)
    Y3 = np.zeros(100) + hauteur_bac
    plt.plot(X3, Y3, color='black')

    # dessin des grains dans le tableau graphique matplot
    couleurs = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown']
    grains = []
    #texts = []
    for grain in range(nb_grains):
        grains.append(ax.add_patch(patches.Circle((POSITION[0, grain, 0], POSITION[0, grain, 1]), radius=rayon, fill=True, color=couleurs[grain%len(couleurs)])))
        #texts.append(ax.text(POSITION[0, grain, 0], POSITION[0, grain, 1], str(grain), ha='center', va='center', fontsize=8, color='white'))
    
    time_text = ax.text(0.05, 0.99, '', transform=ax.transAxes, verticalalignment='top', fontsize=12)
    accelerateur = 10
    def animate(i):
        # Affiche l'indice du temps en haut a gauche de l'écran
        time_text.set_text(f'Indice temps: {i*accelerateur}')
        for grain in range(nb_grains):
            grains[grain].center = (POSITION[i*accelerateur, grain, 0], POSITION[i*accelerateur, grain, 1])
            #texts[grain].set_position((POSITION[i*accelerateur, grain, 0], POSITION[i*accelerateur, grain, 1]))
        return grains + [time_text]
    
    ani = animation.FuncAnimation(fig, animate, frames=int(POSITION.shape[0]/accelerateur), interval=1, blit=True)
    plt.show()





    #-------------------------------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------FONCTIONS-----------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------------------------------#





@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, cache=True)
def allongement_tangentiel_grain_grain(POSITION, VITESSE, i, j, indice_temps, rayon):
    """
    Calcul de l'allongement tangentiel entre deux grains i et j à partir de l'équation

    Paramètres
    ==========
    POSITION : np.array, position des grains
    VITESSE : np.array, vitesse des grains
    i : int, indice du grain i
    j : int, indice du grain j
    indice_temps : int, indice du temps
    rayon : float, rayon des grains
    
    Retour
    ======
    allongement_tangentiel : float, allongement tangentiel entre les grains i et j
    """
    k = 0
    position_i = POSITION[indice_temps-k, i]
    position_j = POSITION[indice_temps-k, j]
    while allongement_normal_grain_grain(position_i, position_j, rayon, rayon) < 0 and k < indice_temps:
        k += 1
        position_i = POSITION[indice_temps-k, i]
        position_j = POSITION[indice_temps-k, j]
    impact = indice_temps - k
    produit_scalaire_array = np.zeros(indice_temps+1) 
    for tps_actuel in range(impact, indice_temps+1):
        position_i = POSITION[tps_actuel,i]
        position_j = POSITION[tps_actuel,j]
        vitesse_i = VITESSE[tps_actuel,i]
        vitesse_j = VITESSE[tps_actuel,j]

        vecteur_normal = (position_i - position_j)/np.linalg.norm(position_i - position_j)
        vecteur_tangent = np.array([-vecteur_normal[1], vecteur_normal[0]])
        if np.dot(vecteur_tangent, vitesse_i) < 0:
            vecteur_tangent = -vecteur_tangent
        vitesse_relative = vitesse_i - vitesse_j
        produit_scalaire = np.dot(vitesse_relative, vecteur_tangent)
        produit_scalaire_array[tps_actuel] = produit_scalaire
        
    # On doit maintenant intégrer la valeur de la vitesse tangentielle sur le temps
    allongement_tangentiel = 0
    for i in range(len(produit_scalaire_array)):
        allongement_tangentiel += produit_scalaire_array[i]
    allongement_tangentiel *= pas_de_temps

    return allongement_tangentiel


@njit(fastmath=True, cache=True)
def actualisation_1(GRILLE, POSITION, VITESSE, VITESSE_DEMI_PAS, ACCELERATION, nb_grains, indice_temps, pas_de_temps, c, largeur_silo_gauche):
    """
    Fonction qui actualise la grille, la position et la vitesse des grains à l'instant k

    Paramètres
    ==========
    GRILLE : np.array, grille contenant les grains
    POSITION : np.array, position des grains
    VITESSE : np.array, vitesse des grains
    nb_grains : int, nombre de grains
    indice_temps : int, indice du temps
    pas_de_temps : float, pas de temps
    c : float, pas d'espace de la grille
    largeur_silo_gauche : float, largeur du silo à gauche

    Retour
    ======
    GRILLE : np.array, grille contenant les grains
    POSITION : np.array, position des grains
    VITESSE : np.array, vitesse des grains
    """

    # Premiere actualisation: position et vitesse à temps k, et grille
    for grain in range(nb_grains):
        # Actualisation position et vitesse
        POSITION[indice_temps] = POSITION[indice_temps-1] + VITESSE_DEMI_PAS[indice_temps-1]*pas_de_temps
        VITESSE[indice_temps] = VITESSE_DEMI_PAS[indice_temps-1] + ACCELERATION[indice_temps-1]*pas_de_temps/2
        # On associe à chaque case de la grille les grains qui sont dans cette case
        #probleme car pos_case peut etre negatif pour ca on déplace le repere:
        try:
            pos_case = (int((POSITION[indice_temps, grain, 0] + abs(largeur_silo_gauche))/c), int((POSITION[indice_temps,grain,1]+ abs(largeur_silo_gauche))/c))
            GRILLE[pos_case[0], pos_case[1], grain] = 1
            if indice_temps == 0:
                print(GRILLE)
        except:
            print("probleme de position")
            print("grain numéro", grain)
            print("indice temps", indice_temps)
            print("position", POSITION[indice_temps, grain])
            print("position case", pos_case)
    
    return GRILLE, POSITION, VITESSE


@njit(fastmath=True)
def voisinage(grain, x, y, GRILLE):
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
            if j == 0 and k == 0:
                for i, bit in enumerate(GRILLE[x,y]):
                    if bit and i != grain:
                        voisinage.append(i)
            else:
                try:
                    for i, bit in enumerate(GRILLE[x+j,y+k]):
                        if bit:
                            voisinage.append(i)
                except:
                    print("probleme de position")
                    print("grain numéro", grain)
                    print("indice temps", indice_temps)
                    print("position", POSITION[indice_temps, grain])
                    print("position case")
                    print("x+j = ", x+j, "y+k = ", y+k)
                    print("x = ", x, "y = ", y)

    return voisinage

@njit(fastmath=True)
def resultante_et_actualisation_2(POSITION, VITESSE, ACCELERATION, VITESSE_DEMI_PAS, GRILLE, nb_grains, indice_temps, pas_de_temps, c, rayon, masse, Agauche, Cgauche, Adroite, Cdroite, largeur_silo_gauche, debut_du_trou, mise_a_jour):
    """
    Fonction qui calcule la force résultante et actualise l'accélération à l'instant k et la vitesse des grains à l'instant k+1/2

    Paramètres
    ==========
    POSITION : np.array, position des grains
    VITESSE : np.array, vitesse des grains
    nb_grains : int, nombre de grains
    indice_temps : int, indice du temps
    pas_de_temps : float, pas de temps
    c : float, pas d'espace de la grille
    rayon : float, rayon des grains
    masse : float, masse des grains
    Agauche : float, coefficient directeur de la paroi gauche
    Cgauche : float, ordonnée à l'origine de la paroi gauche
    Adroite : float, coefficient directeur de la paroi droite
    Cdroite : float, ordonnée à l'origine de la paroi droite
    largeur_silo_gauche : float, largeur du silo à gauche
    debut_du_trou : float, position du début du trou
    mise_a_jour : np.array, liste des grains suceptibles d'être mis à jour

    Retour
    ======
    force_resultante : np.array, force résultante
    ACCELERATION : np.array, accélération des grains
    VITESSE_DEMI_PAS : np.array, vitesse des grains à l'instant k+1/2
    """


    for grain1 in range(nb_grains):
        if mise_a_jour[grain1]:
            force_resultante = np.array([0.0, 0.0])
            force_resultante += application_efforts_distance(masse) #Force à distance = gravité
            # Rencontre avec une paroi du silo ?
            if POSITION[indice_temps, grain1, 1] - rayon >= debut_du_trou:
                # Distance à la paroi, droite d'équation: A*x + B*y + C = 0. Ici B=1, A=-Agauche/droite et C=-Cgauche/droite.
                # La distance es alors donné par la relation : d = abs(A*x + B*y + C) / sqrt(A**2 + B**2)
                distance_a_la_gauche = abs(-Agauche * POSITION[indice_temps, grain1, 0] + 1 * POSITION[indice_temps, grain1, 1] - Cgauche) / np.sqrt(Agauche**2 + 1)
                distance_a_la_droite = abs(-Adroite * POSITION[indice_temps, grain1, 0] + 1 * POSITION[indice_temps, grain1, 1] - Cdroite) / np.sqrt(Adroite**2 + 1)
                penetration_gauche = distance_a_la_gauche - rayon
                penetration_droite = distance_a_la_droite - rayon
                if penetration_gauche < 0:
                    force_resultante += -raideur_mur * penetration_gauche * vecteur_orthogonal_paroi_gauche
                elif penetration_droite < 0:
                    force_resultante += -raideur_mur * penetration_droite * vecteur_orthogonal_paroi_droite
            
            else:
                # Rencontre avec le bac du silo ?
                distance_bac = POSITION[indice_temps, grain1, 1] - hauteur_bac
                penetration_bac = distance_bac - rayon
                if penetration_bac < 0: 
                    #on stope les grains qui sont dans le bac:
                    VITESSE[indice_temps, grain1] = np.array([0,0])
                    ACCELERATION[indice_temps, grain1] = np.array([0,0])
                    VITESSE_DEMI_PAS[indice_temps, grain1] = np.array([0,0])
                    # et on arrete de les mettres a jour:
                    # Trouver l'index de tous les éléments qui ne sont pas égaux à 'grain1'
                    mise_a_jour[grain1] = 0

                    continue


                

            # Rencontre avec un autre grain ?
            position_i_x = POSITION[indice_temps, grain1, 0]
            position_i_y = POSITION[indice_temps, grain1, 1]
            position_i = np.array([position_i_x, position_i_y])
            case_i_x = int((position_i_x + abs(largeur_silo_gauche))/c)
            case_i_y = int((position_i_y + abs(largeur_silo_gauche))/c)
            voisins = voisinage(grain1, case_i_x, case_i_y, GRILLE)
            for grain2 in voisins:
                if mise_a_jour[grain2]:
                    position_j_x = POSITION[indice_temps, grain2, 0]
                    position_j_y = POSITION[indice_temps, grain2, 1]
                    position_j = np.array([position_j_x, position_j_y])
                    if grain1 != grain2:
                        # On définit la force de contact entre les deux grains:
                        force_contact = np.array([0.0, 0.0])
                        allongement_normal = allongement_normal_grain_grain(position_i, position_j, rayon, rayon)
                        # Effort normal
                        if allongement_normal < 0:
                            vecteur_normal = (POSITION[indice_temps, grain1, :] - POSITION[indice_temps, grain2, :])/np.linalg.norm(POSITION[indice_temps, grain1, :] - POSITION[indice_temps, grain2, :])
                            force_contact += -raideur_normale * allongement_normal * vecteur_normal
                            # Effort tangentiel
                            allongement_tangentiel = allongement_tangentiel_grain_grain(POSITION, VITESSE, grain1, grain2, indice_temps, rayon) 
                            vecteur_normal = (POSITION[indice_temps, grain1, :] - POSITION[indice_temps, grain2, :])/np.linalg.norm(POSITION[indice_temps, grain1, :] - POSITION[indice_temps, grain2, :])
                            vecteur_tangentiel = np.array([-vecteur_normal[1], vecteur_normal[0]])
                            if np.dot(vecteur_tangentiel, VITESSE[indice_temps,grain1]) < 0:
                                vecteur_tangentiel = -vecteur_tangentiel
                            force_contact += -raideur_tangentielle * allongement_tangentiel * vecteur_tangentiel
                    
                        # Mise à jour de la résultante des forces sur grain1
                        force_resultante += force_contact

            frotemment = -coefficient_de_frottement * VITESSE[indice_temps, grain1, :]
            force_resultante += frotemment
            # Calcul de l'accélération du grain à partir de l'équation
            ACCELERATION[indice_temps][grain1] = force_resultante / masse
        
            # Calcul de la vitesse de demi-pas à k+1/2 à partir de l'équation
            VITESSE_DEMI_PAS[indice_temps][grain1] = VITESSE_DEMI_PAS[indice_temps-1][grain1] + ACCELERATION[indice_temps][grain1] * pas_de_temps / 2


    return ACCELERATION, VITESSE_DEMI_PAS





#-------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------DEFINITION----------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------------#





# ON PLACE LE REPERE EN BAS A GAUCHE (0,0) DU SILO COIN BAS GAUCHE Y VERS LE HAUT X VERS LA DROITE.
# Définition grain
nb_grains = 100
masse = 2e-3 #kg    
rayon = 1e-2 #m
raideur_normale = 1000 #N/m
raideur_tangentielle = 50 #N/m
coefficient_de_frottement = 0.005 #N/m(air)
# Définir le roulement !


# Définition du temps
temps = 0
indice_temps = 0
pas_de_temps = 1e-3 #s
duree_simulation = 10
nb_temps = int(duree_simulation/pas_de_temps)


# Définition du silo
raideur_mur = 1000 #N/m
limite_bas = -1  #m
limite_haut = 2.5 #m
largeur_silo_gauche = -1 #m
largeur_silo_droite = 1 #m
#debut_du_trou = 0.7 #m en y
# On définit les droites des parois des silos comme des droites de la forme y = Ax + C afin de mettre sous la forme -Ax + y - C = 0
#Agauche, Cgauche = -1/0.6, 0.5
#paroiGauche = lambda x : Agauche*x + Cgauche
#vecteur_directeur_paroi_gauche = np.array([1.0, Agauche])/ np.sqrt(1 + (Agauche)**2) #pointe vers le haut, normalisé
#vecteur_orthogonal_paroi_gauche = np.array([-Agauche, 1.0]) #pointe vers l'intérieur du silo, normalisé
#Adroite, Cdroite = 1/0.6, 0.5
#paroiDroite = lambda x : Adroite*x+Cdroite
#vecteur_directeur_paroi_droite = np.array([1.0, Adroite])/ np.sqrt(1 + (Agauche)**2) #pointe vers le haut, normalisé
#vecteur_orthogonal_paroi_droite = np.array([-Adroite, 1.0]) #pointe vers l'intérieur du silo, normalisé





# TABLEAUX NUMPY
POSITION = np.zeros((nb_temps, nb_grains, 2))   
#Positionnement initiale des grains
i = 0
k = 0
while i < nb_grains:
    for loop in range(20):
        POSITION[0, i, 0] = -0.3 + 3*rayon*loop
        POSITION[0, i, 1] = 2 -k*2*rayon
        i += 1
    k += 1

VITESSE = np.zeros((nb_temps, nb_grains, 2))
VITESSE[0,:,:] = 0 # pas défini au début, on commece à 1 pour la vitesse et à 0 pour la vitessse de demi pas
VITESSE_DEMI_PAS = np.zeros((nb_temps, nb_grains, 2))
VITESSE_DEMI_PAS[0] = np.random.uniform(low=-0.005, high=0.005, size=(nb_grains, 2)) #RAPPEL: Le silo fait un metre par un metre...
ACCELERATION = np.zeros((nb_temps, nb_grains, 2))
ACCELERATION[0,:,:] = 0


# Définition de la grille
c = 5*rayon #pas d'espace de la grille en m
# On définit une grille pour discrétiser l'espace selon le pas d'espace c, a chaque case on met la liste des grains qui sont dans cette case
nb_cases_x = int((largeur_silo_droite - largeur_silo_gauche)/c) + 2
nb_cases_y = int((limite_haut - limite_bas)/c) + 2
GRILLE = np.zeros(( nb_cases_x , nb_cases_y, nb_grains), dtype=int) #on définit une grille vide #ancienne version : GRILLE = {(i,j):[] for i in range(int(largeur_silo_gauche/c)-1, int(largeur_silo_droite/c)+2) for j in range(int(limite_bas/c)-1, int(limite_haut/c)+2)}
#On place les grains dans la grille:
for grain in range(nb_grains):
        # On associe à chaque case de la grille les grains qui sont dans cette case
        #probleme car pos_case peut etre negatif pour ca on déplace le repere:
        pos_case = (int((POSITION[indice_temps,grain,0] + abs(largeur_silo_gauche))/c), int((POSITION[indice_temps,grain,1]+ abs(largeur_silo_gauche))/c))
        GRILLE[pos_case[0], pos_case[1], grain] = 1


mise_a_jour = np.array([1 for i in range(nb_grains)])  #liste qui permet de savoir si on doit mettre à jour le grain ou pas
sorti = 0 #nombre de grains sortis du silo
temps_sortie_debut = 0.0 #temps de sortie du premier grain
temp_sortie_fin = 0.0 #temps de sortie du dernier grain



    #-------------------------------------------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------SIMULATION----------------------------------------------------------------------#
    #-------------------------------------------------------------------------------------------------------------------------------------------#





if __name__ == "__main__":

    app = App()
    app.racine.mainloop()

    Agauche = app.Agauche
    Cgauche = app.Cgauche
    Adroite = app.Adroite
    Cdroite = app.Cdroite
    debut_du_trou = app.debut_du_trou

    paroiGauche = lambda x : Agauche*x + Cgauche
    vecteur_directeur_paroi_gauche = np.array([1.0, Agauche])/ np.sqrt(1 + (Agauche)**2) #pointe vers le haut, normalisé
    vecteur_orthogonal_paroi_gauche = np.array([-Agauche, 1.0]) #pointe vers l'intérieur du silo, normalisé
    paroiDroite = lambda x : Adroite*x+Cdroite
    vecteur_directeur_paroi_droite = np.array([1.0, Adroite])/ np.sqrt(1 + (Agauche)**2) #pointe vers le haut, normalisé
    vecteur_orthogonal_paroi_droite = np.array([-Adroite, 1.0]) #pointe vers l'intérieur du silo, normalisé

    # Definition bac de réception
    hauteur_bac = app.hauteur_bac #m
    largeur_bac_gauche = app.largeur_bac_gauche #m
    largeur_bac_droite = app.largeur_bac_droite #m

    plt.close('all')
    input("Press Enter to continue...")
    start_time = time.time()

    for indice_temps in tqdm(range(1, nb_temps)):
        # Actualisation du temps
        temps += pas_de_temps
        
        #On recrée la grille pour la vider:
        GRILLE = np.zeros(( nb_cases_x , nb_cases_y, nb_grains), dtype=int)

        # Actualisation de la grille, de la position et de la vitesse
        GRILLE, POSITION, VITESSE = actualisation_1(GRILLE, POSITION, VITESSE, VITESSE_DEMI_PAS, ACCELERATION, nb_grains, indice_temps, pas_de_temps, c, largeur_silo_gauche)            
        # Calcul des efforts de contact pour mise à jour des vitesses à temps k+1/2 et accélérations à temps k
        ACCELERATION, VITESSE_DEMI_PAS = resultante_et_actualisation_2(POSITION, VITESSE, ACCELERATION, VITESSE_DEMI_PAS, GRILLE, nb_grains, indice_temps, pas_de_temps, c, rayon, masse, Agauche, Cgauche, Adroite, Cdroite, largeur_silo_gauche, debut_du_trou, mise_a_jour)
    
    # Fin de la boucle principale
    print("Fin de la simulation")
    print("Temps de calcul: ", time.time() - start_time, "secondes")
    #Affichage:
    trajectoire(POSITION, nb_grains, Agauche, Cgauche, Adroite, Cdroite, paroiGauche, paroiDroite, debut_du_trou, hauteur_bac, largeur_bac_gauche, largeur_bac_droite, largeur_silo_gauche, largeur_silo_droite)
    grain_anime(POSITION, nb_grains, rayon, Agauche, Cgauche, Adroite, Cdroite, paroiGauche, paroiDroite, debut_du_trou, hauteur_bac, largeur_bac_gauche, largeur_bac_droite, largeur_silo_gauche, largeur_silo_droite)
