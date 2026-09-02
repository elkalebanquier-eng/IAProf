# IA Prof — professeur virtuel Android hors ligne

IA Prof est une application Android Kotlin qui embarque un véritable modèle génératif GPT-2 quantifié en TensorFlow Lite. Après installation, aucune connexion réseau, API ou serveur n’est nécessaire.

## Modèle intégré

- Modèle : **OpenAI GPT-2 124M, variante TFLite 8 bits `64-8bits.tflite`**
- Source : [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2)
- Licence : MIT
- Taille du modèle : environ **120 Mo**
- Tokenizer intégré : `vocab.json` et `merges.txt` du tokenizer byte-level BPE GPT-2
- Entrée vérifiée : `int32 [1, 64]`
- Sortie principale vérifiée : logits `float32 [1, 64, 50257]`

## Génération

`MainActivity` encode le prompt avec le BPE GPT-2, complète une fenêtre de 64 tokens avec du padding, exécute le modèle TFLite, lit les logits de la dernière position et choisit le token de probabilité maximale. Cette opération est répétée token par token jusqu’à 24 tokens ou jusqu’au token EOS `50256`. Les 12 sorties de cache du fichier TFLite sont également fournies à l’interpréteur pour respecter sa signature multi-sorties.

Le décodage est volontairement glouton (argmax), sans température ni top-k, afin de rester simple et déterministe sur téléphone.

## Compilation

```bash
export ANDROID_HOME="$PWD/sdk"
./gradlew assembleDebug
```

L’APK est produit dans `app/build/outputs/apk/debug/app-debug.apk` et est signé avec la clé debug standard. Android 7 / API 21 ou supérieur est requis.

## Limites

GPT-2 a été entraîné principalement en anglais et n’est pas un modèle spécialisé de tutorat en français. Il peut produire des continuations imparfaites ou incohérentes selon le prompt. Le modèle est réel et génératif, mais sa taille dépasse l’objectif initial de 100 Mo : l’APK final est d’environ 133 Mo en raison du modèle et de la bibliothèque TFLite.

L’application ne déclare aucune permission Internet et tout le traitement est exécuté localement.
