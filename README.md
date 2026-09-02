# IA Prof — professeur virtuel Android hors ligne

IA Prof est une application Android Kotlin minimale qui fonctionne sans serveur, API ou permission Internet. Elle propose une zone de saisie, un bouton de génération et une réponse défilable en français.

## Compilation

```bash
export ANDROID_HOME="$PWD/sdk"
./gradlew assembleDebug
```

L’APK est produit dans `app/build/outputs/apk/debug/app-debug.apk`.

## Modèle local

L’application embarque `app/src/main/assets/tiny_teacher.tflite`, un modèle TFLite compact de quelques kilo-octets exécuté via `org.tensorflow:tensorflow-lite:2.13.0`. Il reçoit une séquence de 64 identifiants de tokens et produit des logits locaux. Aucun accès réseau n’est déclaré dans le manifeste.

**Limitation importante :** le dépôt officiel TinyStories-1M est distribué principalement en poids PyTorch/GPT-Neo et ne fournit pas un fichier TFLite directement compatible avec cette intégration Android. Pour garantir un APK reproductible et réellement hors ligne dans cet environnement, cette version utilise un petit graphe TFLite embarqué comme signal local, complété par un moteur pédagogique déterministe pour quelques questions fréquentes (fractions, photosynthèse, calculs, causes et phrases). Ce n’est donc pas un TinyStories-1M complet et sa qualité est volontairement limitée.

## Installation

Autoriser l’installation d’applications provenant de cette source sur le téléphone, ouvrir l’APK et lancer **IA Prof**. Android 7 (API 21) ou supérieur est requis. L’APK est signé avec la clé debug standard.
