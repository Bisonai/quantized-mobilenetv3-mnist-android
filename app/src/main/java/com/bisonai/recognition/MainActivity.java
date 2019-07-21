/* Copyright 2019 Bisonai Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.bisonai.recognition;

import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.widget.Button;
import android.view.View;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import java.util.List;

import com.bisonai.recognition.classifier.Classifier;
import com.bisonai.recognition.classifier.Classifier.Recognition;
import com.bisonai.recognition.classifier.Classifier.Device;


public abstract class MainActivity extends AppCompatActivity {

    private PaintView paintView;
    private Classifier classifier;
    private TextView classPredictionView;
    private TextView probibilityPredictionView;
    private ProgressBar classificationProgressBar;
    private LinearLayout classAndProbabilityView;
    private Utils utils = new Utils();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        this.paintView = findViewById(R.id.paintView);
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);

        this.classAndProbabilityView = (LinearLayout) findViewById(R.id.classAndProbabilityView);

        this.utils.HideAllViews(this.classAndProbabilityView);

        this.paintView.init(
                metrics,
                (TextView) findViewById(R.id.labelDrawASingleDigit),
                this.classAndProbabilityView
        );

        Button clearButton = findViewById(R.id.clearButton);
        clearButton.setOnClickListener(clearButtonHandler);

        Button classifyButton = findViewById(R.id.classifyButton);
        classifyButton.setOnClickListener(classifyButtonHandler);

        this.classPredictionView = findViewById(R.id.classPredictionView);
        this.probibilityPredictionView = findViewById(R.id.probabilityPredictionView);
        this.classificationProgressBar = findViewById(R.id.classificationProbabilityBar);
        // Set the progress bar to red color
        this.classificationProgressBar.setProgressTintList(ColorStateList.valueOf(Color.RED));

        Device device = Device.CPU;
        int numThreads = -1;
        classifier = createClassifier(device, numThreads);
    }

    View.OnClickListener clearButtonHandler = new View.OnClickListener() {
        public void onClick(View v) {
            paintView.clear();
        }
    };

    View.OnClickListener classifyButtonHandler = new View.OnClickListener() {
        public void onClick(View v) {
            Bitmap bitmap = paintView.getBitmap();
            bitmap = Bitmap.createScaledBitmap(bitmap, classifier.getImageSizeX(), classifier.getImageSizeY(), false);

            List<Recognition> predictions = classifier.recognizeImage(bitmap);
            Recognition top1 = predictions.get(0);

            // Classification visualization
            classPredictionView.setText(top1.getClassName());
            probibilityPredictionView.setText(Float.toString((top1.getConfidence())));
            classificationProgressBar.setProgress(Math.round(top1.getConfidence() * 100));

            utils.ShowAllViews(classAndProbabilityView);
        }
    };

    protected abstract Classifier createClassifier(Device device, int numThreads);
}
