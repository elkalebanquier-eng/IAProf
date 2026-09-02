package com.example.iaprof

import android.graphics.Color
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.Gravity
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    private lateinit var interpreter: Interpreter
    private lateinit var tokenizer: Gpt2Tokenizer
    private lateinit var input: EditText
    private lateinit var chat: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var send: ImageButton
    private val contextLength = 64

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        input = findViewById(R.id.prompt); chat = findViewById(R.id.chat_container)
        scroll = findViewById(R.id.chat_scroll); send = findViewById(R.id.generate)
        findViewById<TextView>(R.id.welcome).movementMethod = ScrollingMovementMethod()
        send.setOnClickListener { generateAsync() }
        input.setOnEditorActionListener { _, _, _ -> generateAsync(); true }
        try {
            tokenizer = Gpt2Tokenizer(this)
            interpreter = Interpreter(loadModel(), Interpreter.Options().apply { setNumThreads(2) })
        } catch (e: Exception) { addAssistant("Le modèle GPT-2 local n’a pas pu être chargé : ${e.message}") }
    }

    private fun loadModel(): ByteBuffer {
        val bytes = assets.open("gpt2/gpt2-64-8bits.tflite").use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
    }

    private fun generateAsync() {
        val prompt = input.text.toString().trim()
        if (prompt.isEmpty() || !::interpreter.isInitialized) return
        addUser(prompt); input.setText(""); send.isEnabled = false
        val thinking = addAssistant("Génération locale…")
        Thread {
            val answer = try { generateText(prompt, 24) } catch (e: Exception) { "Erreur d’inférence locale : ${e.message}" }
            runOnUiThread { thinking.text = answer; send.isEnabled = true; scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) } }
        }.start()
    }

    private fun addUser(text: String) {
        val v = TextView(this).apply { this.text = text; textSize = 16f; setTextColor(Color.WHITE); setPadding(15, 12, 15, 12); setBackgroundResource(R.drawable.user_bubble); gravity = Gravity.START }
        chat.addView(v, LinearLayout.LayoutParams(-2, -2).apply { gravity = Gravity.END; setMargins(42, 12, 0, 0) }); scrollDown()
    }

    private fun addAssistant(text: String): TextView {
        val v = TextView(this).apply { this.text = text; textSize = 16f; setTextColor(Color.rgb(30,41,59)); setLineSpacing(4f, 1f); setPadding(15, 12, 15, 12); setBackgroundResource(R.drawable.assistant_bubble) }
        chat.addView(v, LinearLayout.LayoutParams(-2, -2).apply { gravity = Gravity.START; setMargins(0, 12, 42, 0) }); scrollDown(); return v
    }

    private fun scrollDown() { scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) } }

    private fun generateText(prompt: String, maxTokens: Int): String {
        val ids = tokenizer.encode(prompt).take(contextLength - maxTokens).toMutableList()
        repeat(maxTokens) {
            val window = IntArray(contextLength); val start = maxOf(0, ids.size - contextLength)
            ids.subList(start, ids.size).forEachIndexed { index, id -> window[index] = id }
            val logits = Array(1) { Array(contextLength) { FloatArray(50257) } }
            val outputs = HashMap<Int, Any>(); outputs[0] = logits[0]
            for (i in 1..12) outputs[i] = Array(2) { Array(12) { Array(contextLength) { FloatArray(contextLength) } } }
            interpreter.runForMultipleInputsOutputs(arrayOf(window), outputs)
            val row = logits[0][minOf(ids.size - start - 1, contextLength - 1)]
            var next = 0; var best = row[0]
            for (j in 1 until row.size) if (row[j] > best) { best = row[j]; next = j }
            ids.add(next); if (next == 50256) return tokenizer.decode(ids)
        }
        return tokenizer.decode(ids)
    }

    override fun onDestroy() { if (::interpreter.isInitialized) interpreter.close(); super.onDestroy() }
}
