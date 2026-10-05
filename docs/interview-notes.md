# Questions entretien — irrigation et ML

Problème : prévoir une mesure de sol basse à J+1 pour étudier un composant de screening. Sources NOAA réelles, seuil exploratoire 0,20 m³/m³ ; aucun label irrigation ou diagnostic de stress. Modèles : persistance, logistique et Random Forest. Lire les scores et faux signaux dans reports/metrics.json.

1. **Prédis-tu le besoin réel d'irrigation ?** Non : un événement mesuré de sol superficiel, explicitement défini par seuil.
2. **Pourquoi 0,20 ?** Seuil exploratoire fixe pour rendre le benchmark vérifiable ; il n'est pas universel agronomiquement.
3. **Pourquoi la persistance ?** Le sol change lentement ; ignorer cette baseline surestimerait la valeur du ML.
4. **Que connaît le modèle à la prévision ?** Les mesures historiques jusqu'au jour d'origine, aucune météo future.
5. **Comment gérer les manquants ?** Sentinel converti en NaN, cible exclue si absente, imputation train uniquement pour les entrées.
6. **Pourquoi séparer des stations ?** Tester un transfert géographique plutôt que mémoriser un site.
7. **Comment choisir le seuil de probabilité ?** Grille fixée, F1 validation, jamais test.
8. **Qu'apporte F1 par rapport à accuracy ?** Mettre en évidence l'événement minoritaire et ses faux signaux.
9. **Pourquoi ne pas convertir en mm économisés ?** Profondeur racinaire, capacité du sol, culture et décisions réelles inconnues.
10. **Prochaine étape terrain ?** Mesures racinaires, décisions et consommation observées, holdout ferme, coût des erreurs et seuil validé.

Limites : un seul site indépendant, jours corrélés, couverture saisonnière incomplète et données rétrospectives potentiellement révisées. Développement assisté par IA ; comprendre les unités avant de défendre l'application.
