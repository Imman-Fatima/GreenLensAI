
package com.example.greenlensai;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private Interpreter tflite;

    private ActivityResultLauncher<String> galleryLauncher;
    private ActivityResultLauncher<Void> cameraLauncher;

    private ImageView imagePreview;
    private TextView txtResult, txtConfidence;
    private TextView txtDescription, txtScientificName;
    private TextView txtCategory, txtCare;

    private ArrayList<String> classLabels = new ArrayList<>();

    private static final int IMAGE_SIZE = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(
                findViewById(R.id.main), (v, insets) -> {

                    Insets systemBars = insets.getInsets(
                            WindowInsetsCompat.Type.systemBars()
                    );

                    v.setPadding(
                            systemBars.left,
                            systemBars.top,
                            systemBars.right,
                            systemBars.bottom
                    );

                    return insets;
                }
        );

        imagePreview = findViewById(R.id.imagePreview);

        txtResult = findViewById(R.id.txtResult);
        txtConfidence = findViewById(R.id.txtConfidence);
        txtDescription = findViewById(R.id.txtDescription);
        txtScientificName = findViewById(R.id.txtScientificName);
        txtCategory = findViewById(R.id.txtCategory);
        txtCare = findViewById(R.id.txtCare);

        Button btnGallery = findViewById(R.id.btnGallery);
        Button btnScan = findViewById(R.id.btnScan);
        Button btnHistory = findViewById(R.id.btnHistory);

        loadLabels();
        loadModel();

        // Gallery
        galleryLauncher = registerForActivityResult(
                new ActivityResultContracts.GetContent(),
                uri -> {
                    if (uri != null) {
                        try {
                            InputStream stream =
                                    getContentResolver().openInputStream(uri);

                            Bitmap bitmap = BitmapFactory.decodeStream(stream);

                            if (stream != null) {
                                stream.close();
                            }

                            if (bitmap != null) {
                                imagePreview.setImageBitmap(bitmap);
                                classifyImage(bitmap);
                            }

                        } catch (Exception e) {
                            Toast.makeText(this,
                                    "Image could not be opened",
                                    Toast.LENGTH_SHORT).show();
                        }
                    }
                }
        );

        // Camera
        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicturePreview(),
                bitmap -> {
                    if (bitmap != null) {
                        imagePreview.setImageBitmap(bitmap);
                        classifyImage(bitmap);
                    }
                }
        );

        btnGallery.setOnClickListener(v ->
                galleryLauncher.launch("image/*")
        );

        btnScan.setOnClickListener(v ->
                cameraLauncher.launch(null)
        );

        // History button
        btnHistory.setOnClickListener(v -> showHistory());
    }

    // Load TFLite Model
    private void loadModel() {
        try {
            InputStream inputStream =
                    getAssets().open("GreenLensAI_Final_Model.tflite");

            ByteArrayOutputStream outputStream =
                    new ByteArrayOutputStream();

            byte[] data = new byte[4096];
            int length;

            while ((length = inputStream.read(data)) != -1) {
                outputStream.write(data, 0, length);
            }

            inputStream.close();

            byte[] modelBytes = outputStream.toByteArray();

            ByteBuffer modelBuffer = ByteBuffer.allocateDirect(
                    modelBytes.length
            );

            modelBuffer.order(ByteOrder.nativeOrder());
            modelBuffer.put(modelBytes);
            modelBuffer.rewind();

            tflite = new Interpreter(modelBuffer);

        } catch (Exception e) {
            Toast.makeText(this,
                    "Model loading failed: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    // Load Class Labels
    private void loadLabels() {
        try {
            InputStream inputStream =
                    getAssets().open("GreenLensAI_Class_Labels.json");

            ByteArrayOutputStream outputStream =
                    new ByteArrayOutputStream();

            byte[] data = new byte[1024];
            int length;

            while ((length = inputStream.read(data)) != -1) {
                outputStream.write(data, 0, length);
            }

            inputStream.close();

            String json = outputStream.toString("UTF-8");
            JSONArray array = new JSONArray(json);

            for (int i = 0; i < array.length(); i++) {
                classLabels.add(array.getString(i));
            }

        } catch (Exception e) {
            Toast.makeText(this,
                    "Labels loading failed",
                    Toast.LENGTH_LONG).show();
        }
    }

    // Image Classification
    private void classifyImage(Bitmap bitmap) {

        if (tflite == null || classLabels.size() != 15) {
            Toast.makeText(this,
                    "Model or labels not ready",
                    Toast.LENGTH_SHORT).show();
            return;
        }

        try {
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(
                    bitmap, IMAGE_SIZE, IMAGE_SIZE, true
            );

            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(
                    4 * IMAGE_SIZE * IMAGE_SIZE * 3
            );

            inputBuffer.order(ByteOrder.nativeOrder());

            for (int y = 0; y < IMAGE_SIZE; y++) {
                for (int x = 0; x < IMAGE_SIZE; x++) {

                    int pixel = resizedBitmap.getPixel(x, y);

                    float red = ((pixel >> 16) & 0xFF);
                    float green = ((pixel >> 8) & 0xFF);
                    float blue = (pixel & 0xFF);

                    inputBuffer.putFloat((red / 127.5f) - 1.0f);
                    inputBuffer.putFloat((green / 127.5f) - 1.0f);
                    inputBuffer.putFloat((blue / 127.5f) - 1.0f);
                }
            }

            float[][] output = new float[1][15];

            tflite.run(inputBuffer, output);

            int bestIndex = 0;
            float bestConfidence = output[0][0];

            for (int i = 1; i < 15; i++) {
                if (output[0][i] > bestConfidence) {
                    bestConfidence = output[0][i];
                    bestIndex = i;
                }
            }

            String plantName = classLabels.get(bestIndex);

            displayResult(plantName, bestConfidence);

        } catch (Exception e) {
            Toast.makeText(this,
                    "Classification failed: " + e.getMessage(),
                    Toast.LENGTH_LONG).show();
        }
    }

    // Display Prediction
    private void displayResult(String name, float confidence) {

        saveToHistory(name, confidence);

        txtResult.setText("Result: " + name);

        txtConfidence.setText(
                String.format("Confidence: %.2f%%", confidence * 100)
        );

        txtDescription.setText("Description: " + getDescription(name));

        txtScientificName.setText(
                "Scientific Name: " + getScientificName(name)
        );

        txtCategory.setText("Category: " + getCategory(name));

        txtCare.setText("Care Tips: " + getCareTips(name));
    }

    // Save Scan History
    private void saveToHistory(String name, float confidence) {
        try {
            SharedPreferences preferences =
                    getSharedPreferences("GreenLensAI_History", MODE_PRIVATE);

            String savedHistory =
                    preferences.getString("scan_history", "[]");

            JSONArray historyArray = new JSONArray(savedHistory);

            JSONObject item = new JSONObject();

            item.put("name", name);
            item.put("confidence", confidence * 100);

            item.put("date", new SimpleDateFormat(
                    "dd-MM-yyyy hh:mm a",
                    Locale.getDefault()
            ).format(new Date()));

            historyArray.put(item);

            // Keep latest 100 scans
            if (historyArray.length() > 100) {
                JSONArray latestHistory = new JSONArray();

                for (int i = historyArray.length() - 100;
                     i < historyArray.length(); i++) {

                    latestHistory.put(historyArray.get(i));
                }

                historyArray = latestHistory;
            }

            preferences.edit()
                    .putString("scan_history", historyArray.toString())
                    .apply();

        } catch (Exception e) {
            Toast.makeText(this,
                    "History could not be saved",
                    Toast.LENGTH_SHORT).show();
        }
    }

    // Show Scan History
    private void showHistory() {
        try {
            SharedPreferences preferences =
                    getSharedPreferences("GreenLensAI_History", MODE_PRIVATE);

            String savedHistory =
                    preferences.getString("scan_history", "[]");

            JSONArray historyArray = new JSONArray(savedHistory);

            if (historyArray.length() == 0) {
                new AlertDialog.Builder(this)
                        .setTitle("Scan History")
                        .setMessage("No scan history available yet.")
                        .setPositiveButton("OK", null)
                        .show();
                return;
            }

            StringBuilder historyText = new StringBuilder();

            for (int i = historyArray.length() - 1; i >= 0; i--) {

                JSONObject item = historyArray.getJSONObject(i);

                String name = item.optString("name", "Unknown");
                double confidence = item.optDouble("confidence", 0);
                String date = item.optString("date", "");

                historyText.append("Plant: ").append(name)
                        .append("\nConfidence: ")
                        .append(String.format(
                                Locale.getDefault(),
                                "%.2f%%",
                                confidence
                        ))
                        .append("\nDate: ").append(date)
                        .append("\n\n");
            }

            new AlertDialog.Builder(this)
                    .setTitle("GreenLensAI Scan History")
                    .setMessage(historyText.toString().trim())
                    .setPositiveButton("Close", null)
                    .setNeutralButton("Clear History", (dialog, which) -> {

                        new AlertDialog.Builder(this)
                                .setTitle("Clear History?")
                                .setMessage("Are you sure you want to delete all saved scans?")
                                .setNegativeButton("Cancel", null)
                                .setPositiveButton("Delete", (d, w) -> {

                                    preferences.edit()
                                            .remove("scan_history")
                                            .apply();

                                    Toast.makeText(this,
                                            "History cleared",
                                            Toast.LENGTH_SHORT).show();
                                })
                                .show();
                    })
                    .show();

        } catch (Exception e) {
            Toast.makeText(this,
                    "Could not open scan history",
                    Toast.LENGTH_SHORT).show();
        }
    }

    // Plant Description
    private String getDescription(String name) {
        switch (name) {
            case "AloeVera":
                return "A medicinal succulent plant with thick leaves.";
            case "Aralia":
                return "A decorative indoor plant with attractive foliage.";
            case "Calathea":
                return "A tropical plant known for beautiful patterned leaves.";
            case "KaminiBonsai":
                return "A small ornamental tree commonly grown as bonsai.";
            case "LuckyBamboo":
                return "A popular indoor plant associated with good luck.";
            case "MoneyPlant":
                return "A climbing plant commonly grown indoors.";
            case "Monstera":
                return "A tropical plant with large split leaves.";
            case "PurpleHeart":
                return "A decorative plant with purple-colored leaves.";
            case "SnakePlant":
                return "A hardy indoor plant with upright leaves.";
            case "SpiderPlant":
                return "An indoor plant with long arching leaves.";
            case "daisy":
                return "A flower with a central disc and white petals.";
            case "dandelion":
                return "A yellow flower that develops into a fluffy seed head.";
            case "roses":
                return "A popular flower known for its beautiful petals.";
            case "sunflowers":
                return "A tall flower with a large yellow flower head.";
            case "tulips":
                return "A colorful spring flower with cup-shaped petals.";
            default:
                return "A plant recognized by GreenLensAI.";
        }
    }

    // Scientific Name
    private String getScientificName(String name) {
        switch (name) {
            case "AloeVera": return "Aloe vera";
            case "Aralia": return "Polyscias fruticosa";
            case "Calathea": return "Calathea";
            case "KaminiBonsai": return "Murraya paniculata";
            case "LuckyBamboo": return "Dracaena sanderiana";
            case "MoneyPlant": return "Epipremnum aureum";
            case "Monstera": return "Monstera deliciosa";
            case "PurpleHeart": return "Tradescantia pallida";
            case "SnakePlant": return "Dracaena trifasciata";
            case "SpiderPlant": return "Chlorophytum comosum";
            case "daisy": return "Bellis perennis";
            case "dandelion": return "Taraxacum officinale";
            case "roses": return "Rosa";
            case "sunflowers": return "Helianthus annuus";
            case "tulips": return "Tulipa";
            default: return "Not available";
        }
    }

    // Plant Category
    private String getCategory(String name) {
        switch (name) {
            case "daisy":
            case "dandelion":
            case "roses":
            case "sunflowers":
            case "tulips":
                return "Flower";
            default:
                return "Indoor Plant";
        }
    }

    // Care Tips
    private String getCareTips(String name) {
        switch (name) {
            case "AloeVera":
                return "Provide bright light and water when soil is dry.";
            case "Aralia":
                return "Keep in bright indirect light and water moderately.";
            case "Calathea":
                return "Keep soil moist and provide indirect light.";
            case "KaminiBonsai":
                return "Provide sunlight and water regularly.";
            case "LuckyBamboo":
                return "Keep in indirect light and change water regularly.";
            case "MoneyPlant":
                return "Water when the topsoil becomes dry.";
            case "Monstera":
                return "Provide indirect light and avoid overwatering.";
            case "PurpleHeart":
                return "Provide bright light and well-drained soil.";
            case "SnakePlant":
                return "Water sparingly and keep in indirect light.";
            case "SpiderPlant":
                return "Provide indirect light and moderate watering.";
            case "daisy":
                return "Provide sunlight and water regularly.";
            case "dandelion":
                return "Grow in sunlight and well-drained soil.";
            case "roses":
                return "Provide sunlight, water regularly and prune.";
            case "sunflowers":
                return "Provide full sunlight and regular watering.";
            case "tulips":
                return "Keep in cool conditions and well-drained soil.";
            default:
                return "Provide suitable light and water.";
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (tflite != null) {
            tflite.close();
        }
    }
}
