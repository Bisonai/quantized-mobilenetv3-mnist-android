/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package com.bisonai.recognition.classifier;

import android.app.Activity;

import java.io.IOException;

public class ClassifierQuantizedMobileNet extends Classifier {

    private byte[][] labelProbArray = null;

    public ClassifierQuantizedMobileNet(Activity activity, Device device, int numThreads) throws IOException {
        super(activity, device, numThreads);
        labelProbArray = new byte[1][getNumLabels()];
    }

    @Override
    public int getImageSizeX() {
        return 128;
    }

    @Override
    public int getImageSizeY() {
        return 128;
    }

    @Override
    protected String getModelPath() {
        return "model_mnist.tflite";
    }

    @Override
    protected String getLabelPath() {
        return "labels_mnist.txt";
    }

    @Override
    protected int getNumBytesPerChannel() {
        return 1;
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.put((byte) pixelValue);
    }

    protected int flipValues(int pixelValue) {
        int integerPixelValue = pixelValue & 0xFF;
        return 255 - integerPixelValue;
    }

    @Override
    protected float getNormalizedProbability(int labelIndex) {
        return (labelProbArray[0][labelIndex] & 0xff) / 255.0f;
    }

    @Override
    protected void runInference() {
        tflite.run(imgData, labelProbArray);
    }
}
