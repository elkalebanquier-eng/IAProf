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
    private lateinit var tokenizer: Gpt2Tokenizer
    private lateinit var input: EditText
    private lateinit var output: TextView
    private val contextLength = 64

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        input = findViewById(R.id.prompt)
        output = findViewById(R.id.response)
        output.movementMethod = ScrollingMovementMethod()
        try {
            tokenizer = Gpt2Tokenizer(this)
            interpreter = Interpreter(loadModel(), Interpreter.Options().apply { setNumThreads(2) })
            findViewById<Button>(R.id.generate).setOnClickListener { generateAsync() }
        } catch (e: Exception) {
            output.text = "Impossible de charger le modèle GPT-2 local : ${e.message}"
        }
    }

    private fun loadModel(): ByteBuffer {
        val bytes = assets.open("gpt2/gpt2-64-8bits.tflite").use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
    }

    private fun generateAsync() {
        val prompt = input.text.toString().trim()
        if (prompt.isEmpty()) { output.text = "Écris une question pour commencer."; return }
        val button = findViewById<Button>(R.id.generate)
        button.isEnabled = false; output.text = "Génération locale GPT-2…"
        Thread {
            val answer = generateText(prompt, 24)
            runOnUiThread { output.text = answer; button.isEnabled = true }
        }.start()
    }

    /** Full-sequence autoregressive decoding: GPT-2 receives the current 64-token window at every step. */
    private fun generateText(prompt: String, maxTokens: Int): String {
        val ids = tokenizer.encode(prompt).take(contextLength - maxTokens).toMutableList()
        val originalCount = ids.size
        repeat(maxTokens) {
            val window = IntArray(contextLength)
            val start = maxOf(0, ids.size - contextLength)
            ids.subList(start, ids.size).forEachIndexed { index, id -> window[index] = id }
            val logits = Array(1) { Array(contextLength) { FloatArray(50257) } }
            val caches = HashMap<Int, Any>()
            for (i in 1..12) caches[i] = Array(2) { Array(12) { Array(contextLength) { FloatArray(contextLength) } } }
            val outputs = HashMap<Int, Any>(); outputs[0] = logits[0]
            for (i in 1..12) outputs[i] = caches[i]!!
            interpreter.runForMultipleInputsOutputs(arrayOf(window), outputs)
            val lastIndex = minOf(ids.size - start - 1, contextLength - 1)
            val row = logits[0][lastIndex]
            var next = 0; var best = row[0]
            for (j in 1 until row.size) if (row[j] > best) { best = row[j]; next = j }
            ids.add(next)
            if (next == 50256) return tokenizer.decode(ids)
        }
        return tokenizer.decode(ids.drop(0).take(originalCount + maxTokens))
    }

    override fun onDestroy() { if (::interpreter.isInitialized) interpreter.close(); super.onDestroy() }
}
