package com.skincancerai.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.camera.core.Preview as CameraPreview
import java.util.concurrent.Executors

/**
 * Einstiegspunkt der App: Kamera-Vorschau (CameraX) + Aufnahme + On-Device-Inferenz.
 * Alles läuft lokal auf dem Gerät — keine Cloud, kein Upload von Fotos.
 */
class MainActivity : ComponentActivity() {

    private lateinit var classifier: SkinClassifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        classifier = SkinClassifier(this)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(modifier = Modifier.fillMaxSize()) { AppScreen(classifier) }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier.close()
    }
}

@Composable
fun AppScreen(classifier: SkinClassifier) {
    val context = LocalContext.current
    var hasCamera by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = androidx.activity.compose.rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasCamera = granted }

    LaunchedEffect(Unit) {
        if (!hasCamera) launcher.launch(Manifest.permission.CAMERA)
    }

    var result by remember { mutableStateOf<SkinClassifier.Result?>(null) }
    val imageCapture = remember { ImageCapture.Builder().build() }
    val executor = remember { Executors.newSingleThreadExecutor() }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp).verticalScroll(rememberScrollState())) {
        Text("SkinCancerAI", style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(8.dp))

        DisclaimerCard()
        Spacer(Modifier.height(12.dp))

        if (hasCamera) {
            CameraPreviewView(imageCapture, Modifier.fillMaxWidth().height(360.dp))
            Spacer(Modifier.height(12.dp))
            Button(
                onClick = {
                    imageCapture.takePicture(executor,
                        object : ImageCapture.OnImageCapturedCallback() {
                            override fun onCaptureSuccess(image: ImageProxy) {
                                val bmp = image.toBitmap()
                                image.close()
                                result = classifier.classify(bmp)
                            }
                            override fun onError(exc: ImageCaptureException) { /* Fehler-UI hier */ }
                        })
                },
                modifier = Modifier.fillMaxWidth()
            ) { Text("Foto analysieren (aus ~15–20 cm Abstand)") }
        } else {
            Text("Kamerazugriff wird für die Analyse benötigt.")
        }

        Spacer(Modifier.height(16.dp))
        result?.let { ResultCard(it) }
    }
}

@Composable
fun CameraPreviewView(imageCapture: ImageCapture, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = androidx.compose.ui.platform.LocalLifecycleOwner.current
    AndroidView(
        modifier = modifier.clip(RoundedCornerShape(16.dp)),
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val providerFuture = ProcessCameraProvider.getInstance(ctx)
            providerFuture.addListener({
                val provider = providerFuture.get()
                val preview = CameraPreview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture
                )
            }, ContextCompat.getMainExecutor(ctx))
            previewView
        }
    )
}

@Composable
fun ResultCard(result: SkinClassifier.Result) {
    val pct = (result.malignantProbability * 100).toInt()
    val container = if (result.suspicious) Color(0xFF7F1D1D) else Color(0xFF14532D)
    Card(colors = CardDefaults.cardColors(containerColor = container),
        modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text(
                if (result.suspicious) "⚠️  Auffällig — ärztlich abklären lassen"
                else "Unauffällig (kein Ersatz für ärztliche Kontrolle)",
                style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(6.dp))
            Text("Modell-Einschätzung „verdächtig": $pct %")
            Spacer(Modifier.height(6.dp))
            Text(
                "Dies ist KEINE Diagnose. Suche bei jeder auffälligen oder sich " +
                        "verändernden Hautstelle eine dermatologische Praxis auf.",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
fun DisclaimerCard() {
    Card(colors = CardDefaults.cardColors(containerColor = Color(0xFF3F3F46)),
        modifier = Modifier.fillMaxWidth()) {
        Text(
            "⚠️ Kein Medizinprodukt. Nur für helle Hauttypen (Fitzpatrick I–III) " +
                    "als Demo trainiert. Ergebnisse ersetzen keine ärztliche Untersuchung.",
            modifier = Modifier.padding(12.dp),
            style = MaterialTheme.typography.bodySmall
        )
    }
}
