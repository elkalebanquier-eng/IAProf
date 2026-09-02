package com.iaprof.app

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.viewModelScope
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) { super.onCreate(savedInstanceState); setContent { IAProfApp() } }
}

data class ChatMessage(val text: String, val user: Boolean)
enum class ModelStatus { NOT_FOUND, LOADING, READY, GENERATING, ERROR }
data class UiState(val status: ModelStatus = ModelStatus.NOT_FOUND, val modelName: String? = null, val messages: List<ChatMessage> = emptyList(), val error: String? = null)

class MainViewModel : ViewModel() {
    private val _state = MutableStateFlow(UiState(messages = listOf(ChatMessage("Sélectionne un modèle .task ou .bin local pour commencer. Rien ne sera envoyé sur Internet.", false))))
    val state: StateFlow<UiState> = _state.asStateFlow()
    private var engine: LlmInference? = null
    private var modelFile: File? = null
    private val maxContext = 512

    fun loadBundledIfPresent(context: Context) {
        viewModelScope.launch(Dispatchers.Default) {
            if (!context.assets.list("").orEmpty().contains("model.task")) return@launch
            _state.value = _state.value.copy(status = ModelStatus.LOADING, error = null)
            try {
                val file = File(context.filesDir, "bundled-model.task")
                context.assets.open("model.task").use { input -> file.outputStream().use { output -> input.copyTo(output) } }
                closeEngine(); engine = createEngine(context, file.absolutePath, preferGpu = true); modelFile = file
                _state.value = _state.value.copy(status = ModelStatus.READY, modelName = "model.task (embarqué)")
            } catch (t: Throwable) { closeEngine(); _state.value = _state.value.copy(status = ModelStatus.ERROR, error = friendlyError(t)) }
        }
    }

    fun loadFromUri(context: Context, uri: Uri) {
        viewModelScope.launch(Dispatchers.Default) {
            _state.value = _state.value.copy(status = ModelStatus.LOADING, error = null)
            try {
                val file = File(context.filesDir, "local-model.task")
                context.contentResolver.openInputStream(uri)?.use { input -> file.outputStream().use { output -> input.copyTo(output) } } ?: error("Fichier inaccessible")
                if (file.length() < 1024) error("Le fichier est trop petit ou corrompu")
                val sizeMb = file.length() / 1024 / 1024
                if (sizeMb > 4096) error("Modèle trop volumineux pour la sécurité mémoire (>${4096} Mo)")
                closeEngine()
                engine = createEngine(context, file.absolutePath, preferGpu = true)
                modelFile = file
                _state.value = _state.value.copy(status = ModelStatus.READY, modelName = uri.lastPathSegment ?: "modèle local")
            } catch (t: Throwable) { closeEngine(); _state.value = _state.value.copy(status = ModelStatus.ERROR, error = friendlyError(t)) }
        }
    }

    private fun createEngine(context: Context, path: String, preferGpu: Boolean): LlmInference {
        val backend = if (preferGpu) LlmInference.Backend.GPU else LlmInference.Backend.CPU
        val options = LlmInference.LlmInferenceOptions.builder().setModelPath(path).setMaxTokens(maxContext).setMaxTopK(40).setPreferredBackend(backend).build()
        return LlmInference.createFromOptions(context, options)
    }

    fun send(prompt: String) {
        val clean = prompt.trim(); if (clean.isEmpty() || engine == null || _state.value.status == ModelStatus.GENERATING) return
        _state.value = _state.value.copy(status = ModelStatus.GENERATING, messages = _state.value.messages + ChatMessage(clean, true) + ChatMessage("", false), error = null)
        viewModelScope.launch(Dispatchers.Default) {
            try {
                engine?.generateResponseAsync(clean) { partial, done ->
                    val current = _state.value.messages.toMutableList(); if (current.isNotEmpty()) current[current.lastIndex] = ChatMessage(partial, false)
                    _state.value = _state.value.copy(messages = current, status = if (done) ModelStatus.READY else ModelStatus.GENERATING)
                }
            } catch (t: Throwable) {
                // GPU incompatibility is handled by recreating the same local file on CPU.
                try { closeEngine(); engine = createEngine(AppContextHolder.context, modelFile!!.absolutePath, preferGpu = false); engine?.generateResponseAsync(clean) { partial, done -> updateLast(partial, done) } }
                catch (fallback: Throwable) { updateLast("Erreur hors ligne : ${friendlyError(fallback)}", true) }
            }
        }
    }

    private fun updateLast(text: String, done: Boolean) { val list = _state.value.messages.toMutableList(); if (list.isNotEmpty()) list[list.lastIndex] = ChatMessage(text, false); _state.value = _state.value.copy(messages = list, status = if (done) ModelStatus.READY else ModelStatus.GENERATING) }
    private fun friendlyError(t: Throwable) = when { t.message?.contains("memory", true) == true -> "Mémoire RAM insuffisante. Choisis un modèle quantifié plus léger."; t.message?.contains("model", true) == true -> "Modèle invalide ou non compatible MediaPipe."; else -> (t.message ?: "Erreur inconnue").take(180) }
    fun closeEngine() { try { engine?.close() } catch (_: Throwable) {}; engine = null }
    override fun onCleared() { closeEngine(); super.onCleared() }
}

private object AppContextHolder { lateinit var context: Context }

@OptIn(ExperimentalMaterial3Api::class)
@Composable fun IAProfApp(vm: MainViewModel = viewModel()) {
    val context = androidx.compose.ui.platform.LocalContext.current; AppContextHolder.context = context.applicationContext
    LaunchedEffect(Unit) { vm.loadBundledIfPresent(context) }
    val state by vm.state.collectAsState(); var text by remember { mutableStateOf("") }; val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { it?.let { uri -> try { context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: SecurityException) {}; vm.loadFromUri(context, uri) } }
    MaterialTheme(colorScheme = lightColorScheme(primary = Color(0xFF635BFF))) { Scaffold(topBar = { TopAppBar(title = { Text("IA Prof", fontWeight = FontWeight.Bold) }, actions = { Text("HORS-LIGNE", color = Color(0xFF15803D), style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(end = 16.dp)) }) }) { pad -> Column(Modifier.padding(pad).fillMaxSize().background(Color(0xFFF8FAFC))) {
        StatusCard(state, onChoose = { picker.launch(arrayOf("application/octet-stream", "application/*")) })
        val listState = rememberLazyListState(); LaunchedEffect(state.messages.size, state.messages.lastOrNull()?.text) { if (state.messages.isNotEmpty()) listState.animateScrollToItem(state.messages.lastIndex) }
        LazyColumn(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 14.dp), state = listState, contentPadding = PaddingValues(vertical = 12.dp)) { items(state.messages) { msg -> Bubble(msg) } }
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.Bottom) { OutlinedTextField(text, { text = it }, Modifier.weight(1f), placeholder = { Text("Écris ton message…") }, maxLines = 4); IconButton(onClick = { vm.send(text); text = "" }, enabled = state.status == ModelStatus.READY) { Icon(Icons.Default.Send, "Envoyer") } }
    } } }
}

@Composable private fun StatusCard(state: UiState, onChoose: () -> Unit) { Card(Modifier.fillMaxWidth().padding(12.dp), colors = CardDefaults.cardColors(containerColor = Color.White)) { Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) { Column(Modifier.weight(1f)) { Text(when (state.status) { ModelStatus.NOT_FOUND -> "Modèle non trouvé"; ModelStatus.LOADING -> "Chargement du modèle en RAM…"; ModelStatus.READY -> "Connecté au modèle local"; ModelStatus.GENERATING -> "Génération…"; ModelStatus.ERROR -> "Erreur de modèle" }, fontWeight = FontWeight.Bold); Text(state.modelName ?: state.error ?: "Aucun modèle chargé", style = MaterialTheme.typography.bodySmall, color = Color.Gray) }; if (state.status == ModelStatus.NOT_FOUND || state.status == ModelStatus.ERROR) Button(onClick = onChoose) { Icon(Icons.Default.AttachFile, null); Spacer(Modifier.width(4.dp)); Text("Choisir") } else if (state.status == ModelStatus.LOADING) CircularProgressIndicator(Modifier.size(24.dp)) else Text("LOCAL", color = Color(0xFF15803D), style = MaterialTheme.typography.labelSmall) } } }
@Composable private fun Bubble(msg: ChatMessage) { Row(Modifier.fillMaxWidth(), horizontalArrangement = if (msg.user) Arrangement.End else Arrangement.Start) { Surface(color = if (msg.user) Color(0xFF635BFF) else Color.White, shape = RoundedCornerShape(18.dp), tonalElevation = 1.dp, modifier = Modifier.widthIn(max = 310.dp).padding(vertical = 5.dp)) { Text(msg.text.ifEmpty { "…" }, color = if (msg.user) Color.White else Color(0xFF1E293B), modifier = Modifier.padding(14.dp)) } } }
