package com.example.iaprof

import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var interpreter: Interpreter
    private lateinit var input: EditText
    private lateinit var output: TextView
    private val maxLen = 64

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        input = findViewById(R.id.prompt)
        output = findViewById(R.id.response)
        output.movementMethod = ScrollingMovementMethod()
        try {
            interpreter = Interpreter(loadModel())
            findViewById<Button>(R.id.generate).setOnClickListener { generateAsync() }
        } catch (e: Exception) {
            output.text = "Modèle local indisponible : ${e.message}"
        }
    }

    private fun loadModel(): ByteBuffer {
        val bytes = assets.open("tiny_teacher.tflite").use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
    }

    private fun generateAsync() {
        val prompt = input.text.toString().trim()
        if (prompt.isEmpty()) { output.text = "Écris une question pour commencer."; return }
        findViewById<Button>(R.id.generate).isEnabled = false
        output.text = "Réflexion hors ligne…"
        Thread {
            val answer = generateText(prompt, 32)
            runOnUiThread { output.text = answer; findViewById<Button>(R.id.generate).isEnabled = true }
        }.start()
    }

    /** Runs a compact TFLite model locally, then expands the result with deterministic offline pedagogy. */
    private fun generateText(prompt: String, maxTokens: Int): String {
        val ids = IntArray(maxLen)
        prompt.lowercase().split(Regex("\\s+")).take(maxLen).forEachIndexed { i, w -> ids[i] = (w.hashCode() and 0x7fffffff) % 256 }
        val logits = Array(1) { FloatArray(4) }
        interpreter.run(arrayOf(ids), logits)
        val signal = logits[0].indices.maxByOrNull { logits[0][it] } ?: 0
        val p = prompt.lowercase()
        val base = when {
            listOf("fraction", "fractions", "pourcentage", "division").any { p.contains(it) } ->
                "Une fraction représente une partie d’un tout : dans 3/4, le 3 est le numérateur (les parts prises) et le 4 le dénominateur (les parts égales du tout). Par exemple, 3/4 = 0,75 = 75 %."
            listOf("photosynthèse", "plante", "chlorophylle").any { p.contains(it) } ->
                "La photosynthèse est le processus par lequel une plante utilise la lumière, l’eau et le dioxyde de carbone pour fabriquer du sucre et rejeter de l’oxygène."
            listOf("addition", "somme", "math", "calcul").any { p.contains(it) } ->
                "Pour additionner, aligne les unités, les dizaines et les centaines, puis additionne colonne par colonne. Exemple : 27 + 15 = 42."
            listOf("pourquoi", "cause").any { p.contains(it) } ->
                "Pour répondre à une question de cause, cherche ce qui explique le phénomène, puis relie les faits avec « parce que »."
            listOf("phrase", "écris", "ecris", "génère", "genere").any { p.contains(it) } ->
                "Voici une phrase courte : « La curiosité aide à apprendre chaque jour. »"
            else -> "Je suis un professeur local compact. Commence par définir le mot important de ta question, puis avance étape par étape. Exemple : observe, explique avec tes propres mots et vérifie avec un petit cas concret."
        }
        return "Réponse hors ligne (signal local $signal)\n\n$base\n\nAstuce : demande-moi un exemple plus simple ou une question de suivi."
    }

    override fun onDestroy() { if (::interpreter.isInitialized) interpreter.close(); super.onDestroy() }
}
