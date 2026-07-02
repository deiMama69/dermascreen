package com.skincancerai.app

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * On-Device-Inferenz mit dem exportierten TFLite-Modell.
 *
 * Das Modell gibt EINE Wahrscheinlichkeit p (maligne) im Bereich [0,1] aus. Die
 * Normalisierung (EfficientNet preprocess_input) ist bereits IM Modell eingebettet,
 * daher wird das Bitmap nur auf die Eingabegröße skaliert und als Float [0,255]
 * übergeben — keine zusätzliche Normalisierung nötig.
 *
 * Der Betriebs-Schwellenwert stammt aus model_meta.json (in der Evaluierung auf
 * eine Ziel-Sensitivität kalibriert). Liegt p darüber, gilt die Läsion als
 * "verdächtig".
 *
 * ⚠️ Kein Medizinprodukt. Das Ergebnis ist keine Diagnose.
 */
class SkinClassifier(context: Context) {

    private val interpreter: Interpreter
    private val inputSize: Int
    val threshold: Float

    init {
        val model = FileUtil.loadMappedFile(context, MODEL_FILE)
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(4) })

        val metaJson = context.assets.open(META_FILE).bufferedReader().use { it.readText() }
        val meta = JSONObject(metaJson)
        inputSize = meta.optInt("input_size", 224)
        threshold = meta.optDouble("operating_threshold", 0.5).toFloat()
    }

    data class Result(val malignantProbability: Float, val suspicious: Boolean)

    fun classify(bitmap: Bitmap): Result {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val input = toFloatBuffer(resized)
        val output = Array(1) { FloatArray(1) }
        interpreter.run(input, output)
        val p = output[0][0]
        return Result(malignantProbability = p, suspicious = p >= threshold)
    }

    private fun toFloatBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in pixels) {
            // Rohe RGB-Werte [0,255]; preprocess_input steckt im Modell.
            buffer.putFloat(((pixel shr 16) and 0xFF).toFloat())
            buffer.putFloat(((pixel shr 8) and 0xFF).toFloat())
            buffer.putFloat((pixel and 0xFF).toFloat())
        }
        buffer.rewind()
        return buffer
    }

    fun close() = interpreter.close()

    companion object {
        private const val MODEL_FILE = "skincancer.tflite"
        private const val META_FILE = "model_meta.json"
    }
}
