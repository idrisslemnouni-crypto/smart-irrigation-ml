# Comprendre le screening d'humidité

Une sonde mesure une teneur en eau volumique : 0,20 m³/m³ n'est pas 20 mm d'eau ni un besoin d'irrigation. L'étiquette du projet est explicitement « mesure demain sous 0,20 », un seuil exploratoire défini avant les modèles. Elle ne remplace pas une décision de l'agriculteur ou un seuil physiologique validé. Les sites NOAA ne sont pas décrits comme des parcelles irriguées.

La persistance prévoit que l'état d'aujourd'hui continuera demain. Il faut la battre pour montrer un intérêt du ML. Le modèle utilise les mesures jusqu'à la fin du jour t, des moyennes des sept jours passés et la saison. Le calendrier doit être continu avant shift : sinon le jour suivant pourrait signifier une semaine plus tard. Les valeurs manquantes ne deviennent pas de faux négatifs et les cibles ne sont pas imputées.

Le test mélange deux exigences : une date future et une station jamais vue. Les données Illinois sont exclues de l'entraînement et de la validation. Standardisation et imputation apprennent sur train seulement. Le seuil de probabilité est choisi sur validation ; regarder le test pour le modifier ferait perdre le test indépendant.

F1 résume précision et rappel de l'événement rare. PR-AUC mesure le classement sur différents seuils. La matrice expose les faux signaux et les événements manqués : le RF détecte davantage d'événements mais émet aussi davantage de faux signaux. Une meilleure F1 n'établit donc pas automatiquement une meilleure décision.

Les variables de sol sont corrélées : la permutation peut répartir leur importance. La série temporelle et l'absence de mesure durant certaines périodes limitent l'indépendance et la représentativité. À apprendre : refaire un label à partir d'une vraie ligne, montrer les dates utilisées pour une moyenne mobile, expliquer une fausse alarme, puis distinguer cette alerte d'un bilan hydrique racinaire. Une amélioration utile nécessite des données terrain et un coût de décision, pas seulement plus d'arbres.
