# IA Prof — professeur virtuel Android hors ligne

IA Prof est une application Android Kotlin avec une interface de conversation inspirée de ChatGPT et Manus. Chaque réponse de l’assistant est générée par le modèle GPT-2 local embarqué : il n’y a pas de réponses prédéfinies, de backend ou d’appel réseau.

## Modèle intégré

- Modèle : **OpenAI GPT-2 124M, variante TFLite 8 bits `64-8bits.tflite`**
- Source : [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2)
- Licence : MIT
- Taille du modèle : environ **120 Mo**
- Tokenizer intégré : `vocab.json` et `merges.txt` du tokenizer byte-level BPE GPT-2
- Entrée vérifiée : `int32 [1, 64]`
- Sortie principale vérifiée : logits `float32 [1, 64, 50257]`

## Génération réelle

Pour chaque message, l’application encode le texte avec le BPE GPT-2, exécute le modèle TFLite sur la fenêtre courante, lit les logits de la dernière position et choisit le token suivant par argmax. La boucle est répétée token par token jusqu’à 24 tokens ou EOS. La conversation affiche séparément les messages utilisateur et les réponses générées, avec défilement automatique.

## Hors ligne et confidentialité

Le modèle, le tokenizer et le moteur d’inférence sont inclus dans l’APK. Le manifeste ne déclare aucune permission Internet. Les messages restent dans la mémoire de l’écran pendant la session et ne sont envoyés à aucun serveur.

## Compilation

```bash
export ANDROID_HOME="$PWD/sdk"
./gradlew assembleDebug
```

L’APK est signé avec la clé debug standard et cible Android 7 / API 21 ou supérieur. La taille finale est d’environ 133 Mo en raison du modèle GPT-2.

## Limites

GPT-2 a été entraîné principalement en anglais et n’est pas spécialisé dans le tutorat en français. Ses réponses peuvent être imparfaites ou incohérentes. Le modèle est toutefois un vrai modèle génératif : les textes produits ne proviennent pas d’une liste de réponses codées en dur.
