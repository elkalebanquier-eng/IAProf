# IA Prof 3.0 — assistant local Android

IA Prof est une application Android **100 % hors ligne** écrite en Kotlin et Jetpack Compose. Elle utilise l’API MediaPipe LLM Inference (`tasks-genai:0.10.27`) pour exécuter un modèle génératif local compatible `.task` ou `.bin`.

## Fonctionnalités

L’interface propose un chat moderne avec messages utilisateur/assistant, états `Modèle non trouvé`, `Chargement du modèle en RAM…`, `Prêt` et `Génération…`, ainsi qu’un sélecteur Storage Access Framework pour choisir un modèle présent sur le téléphone. Si un fichier `model.task` est placé dans `app/src/main/assets`, il est chargé automatiquement ; sinon l’utilisateur peut choisir un fichier local.

Les réponses sont reçues en streaming via `generateResponseAsync`. Le moteur est créé sur `Dispatchers.Default`, utilise une fenêtre maximale de 512 tokens, tente le backend GPU puis recrée automatiquement le moteur en CPU si le GPU n’est pas compatible. Les erreurs de fichier invalide et de mémoire sont affichées dans l’interface sans faire tomber l’application. `LlmInference.close()` est appelé lors du remplacement du modèle et dans `ViewModel.onCleared()`.

## Modèle

Le projet ne télécharge aucun modèle et ne fait aucune vérification de licence en ligne. Il est volontairement livré sans gros modèle dans `assets`, car les modèles Gemma/Llama `.task` font souvent plusieurs centaines de Mo et dépassent les limites pratiques d’un APK. L’utilisateur peut placer son modèle MediaPipe compatible en asset ou le sélectionner localement. Exemples : Gemma 2B IT INT4, Gemma 3 1B ou Llama 3.2 1B dans un format pris en charge par MediaPipe.

## Compilation

```bash
export ANDROID_HOME="$PWD/sdk"
./gradlew assembleDebug
```

Package : `com.iaprof.app`. Android minimum : API 24. Le manifeste ne déclare aucune permission Internet. Le fichier produit est `app/build/outputs/apk/debug/app-debug.apk`.

## Note API

La documentation Google indique que MediaPipe LLM Inference est désormais en maintenance et recommande LiteRT-LM pour les nouveaux projets. Cette version conserve MediaPipe afin de respecter le cahier des charges demandé et d’exposer directement `LlmInferenceOptions`, le backend GPU/CPU et le streaming Android.
