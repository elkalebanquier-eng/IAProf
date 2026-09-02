package com.example.iaprof

import android.content.Context
import org.json.JSONObject
import java.nio.charset.StandardCharsets

class Gpt2Tokenizer(context: Context) {
    private val vocab = HashMap<String, Int>()
    private val idToToken = HashMap<Int, String>()
    private val ranks = HashMap<String, Int>()
    private val byteToUnicode = HashMap<Int, Char>()
    private val unicodeToByte = HashMap<Char, Int>()
    private val cache = HashMap<String, List<String>>()
    private val pattern = Regex("'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\\d+| ?[^\\sA-Za-z\\d]+|\\s+(?!\\s)|\\s+")

    init {
        val vocabJson = context.assets.open("gpt2/vocab.json").bufferedReader().use { it.readText() }
        val obj = JSONObject(vocabJson)
        for (key in obj.keys()) { val id = obj.getInt(key); vocab[key] = id; idToToken[id] = key }
        context.assets.open("gpt2/merges.txt").bufferedReader().useLines { lines ->
            lines.drop(1).forEachIndexed { i, line -> if (line.isNotBlank()) ranks[line] = i }
        }
        buildByteMap()
    }

    private fun buildByteMap() {
        // GPT-2's byte-level alphabet: printable bytes retain their code point;
        // the remaining bytes are mapped into the private Unicode range.
        val keep = (33..126).toMutableList().apply { addAll(161..172); addAll(174..255) }
        val extra = (0..255).filter { it !in keep }
        keep.forEach { b -> byteToUnicode[b] = b.toChar(); unicodeToByte[b.toChar()] = b }
        extra.forEachIndexed { i, b -> val c = (256 + i).toChar(); byteToUnicode[b] = c; unicodeToByte[c] = b }
    }

    fun encode(text: String): MutableList<Int> {
        val ids = mutableListOf<Int>()
        pattern.findAll(text).forEach { match ->
            val transformed = match.value.toByteArray(StandardCharsets.UTF_8).joinToString("") { byteToUnicode[it.toInt() and 255].toString() }
            bpe(transformed).forEach { token -> ids.add(vocab[token] ?: 0) }
        }
        return ids
    }

    private fun bpe(token: String): List<String> {
        cache[token]?.let { return it }
        if (token.length <= 1) return listOf(token)
        val symbols = token.map { it.toString() }.toMutableList()
        while (true) {
            var best: String? = null; var bestRank = Int.MAX_VALUE
            for (i in 0 until symbols.size - 1) {
                val pair = symbols[i] + " " + symbols[i + 1]
                val r = ranks[pair] ?: continue
                if (r < bestRank) { bestRank = r; best = pair }
            }
            if (best == null) break
            val parts = best.split(' '); val merged = mutableListOf<String>(); var i = 0
            while (i < symbols.size) {
                if (i < symbols.size - 1 && symbols[i] == parts[0] && symbols[i + 1] == parts[1]) { merged.add(parts[0] + parts[1]); i += 2 } else { merged.add(symbols[i]); i++ }
            }
            symbols.clear(); symbols.addAll(merged)
        }
        cache[token] = symbols
        return symbols
    }

    fun decode(ids: List<Int>): String {
        val chars = StringBuilder()
        ids.forEach { id -> idToToken[id]?.forEach { c -> unicodeToByte[c]?.let { chars.append(it.toChar()) } } }
        return String(chars.toString().toByteArray(Charsets.ISO_8859_1), StandardCharsets.UTF_8)
    }
}
