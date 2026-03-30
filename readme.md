Parameter Learning in Dynamical Systems

Ce dépôt contient les codes et les résultats d'un projet de recherche portant sur le développement de modèles basés sur les données pour la résolution de problèmes inverses. L'objectif principal est d'apprendre les paramètres de systèmes dynamiques (notamment le système de Lorenz-63 en régime chaotique) à partir d'observations bruitées.
📋 Table des matières

    Description du projet

    Structure du dépôt

    Méthodes explorées

    Auteur

🔬 Description du projet

L'identification des paramètres de systèmes dynamiques non linéaires et chaotiques est un défi majeur, en particulier lorsque les données d'observation sont bruitées. Ce projet explore, compare et implémente différentes approches numériques et d'apprentissage automatique (Machine Learning) pour reconstruire la dynamique et estimer les paramètres du système :

    Réseaux de Neurones Informés par la Physique (PINNs)

    Algorithme SINDy (Sparse Identification of Nonlinear Dynamics) dans sa formulation faible (Weak SINDy)

    Optimisation par Descente de Gradient Stochastique (SGD)

📂 Structure du dépôt

Le projet est organisé en trois dossiers principaux, correspondant aux trois méthodes étudiées :

    📁 PINNs/ : Contient l'implémentation des Physics-Informed Neural Networks.

        pinn_t3_noise.ipynb : Notebook Jupyter principal pour l'entraînement du modèle sur des données bruitées.

        outputs PINNs/ : Dossiers contenant les graphiques de résultats (évolution des pertes, trajectoires prédites vs réelles, convergence des paramètres).

    📁 Sindy/ : Implémentation de la méthode SINDy.

        Weak_Sindy.py : Script Python pour l'identification clairsemée de la dynamique sous forme faible.

        weakSINdy_noised.png : Aperçu des résultats sur des données bruitées.

    📁 sgd/ : Approche d'optimisation classique.

        sgd.py / sgd.ipynb : Scripts et notebook pour l'optimisation par descente de gradient stochastique.

        utils.py : Fonctions utilitaires pour le modèle SGD.

        config.ini : Fichier de configuration des hyperparamètres.

    📄 Soutenance.pdf : Support de présentation résumant les travaux et les résultats du projet.

⚙️ Méthodes explorées
1. Physics-Informed Neural Networks (PINNs)

Les PINNs intègrent les équations différentielles régissant le système (ex: Lorenz-63) directement dans la fonction de perte (loss function) du réseau de neurones. Cela permet au modèle de respecter la physique du système tout en apprenant les paramètres inconnus à partir d'observations bruitées.
2. Weak SINDy

Une adaptation de la méthode SINDy classique. La formulation faible (Weak formulation) permet d'atténuer l'impact du bruit de mesure en intégrant les données sur des fonctions tests, évitant ainsi le calcul numérique explicite des dérivées, particulièrement instable en présence de bruit.
3. Stochastic Gradient Descent (SGD)

Une approche d'optimisation standard visant à minimiser l'erreur de reconstruction de la trajectoire en ajustant itérativement les paramètres du système. Le fichier config.ini permet de paramétrer finement l'algorithme.
👤 Auteurs

Gautier Tandeau de Marsac, Houcine Ayoubi, Victor Martin