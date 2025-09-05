/*
 * Copyright 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.onnxruntime.example.imageclassifier;

import android.graphics.Bitmap;
import java.nio.FloatBuffer;

public class ImageUtil {
    public static final int DIM_BATCH_SIZE = 1;
    public static final int DIM_PIXEL_SIZE = 3;
    public static final int IMAGE_SIZE_X = 320; // NudeNet 320n model expects 320x320
    public static final int IMAGE_SIZE_Y = 320;

    public static FloatBuffer preProcess(Bitmap bitmap) {
        FloatBuffer imgData = FloatBuffer.allocate(
                DIM_BATCH_SIZE
                        * DIM_PIXEL_SIZE
                        * IMAGE_SIZE_X
                        * IMAGE_SIZE_Y
        );
        imgData.rewind();
        int stride = IMAGE_SIZE_X * IMAGE_SIZE_Y;
        int[] bmpData = new int[stride];
        bitmap.getPixels(bmpData, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        
        for (int i = 0; i < IMAGE_SIZE_X; i++) {
            for (int j = 0; j < IMAGE_SIZE_Y; j++) {
                int idx = IMAGE_SIZE_Y * i + j;
                int pixelValue = bmpData[idx];
                imgData.put(idx, (((pixelValue >> 16 & 0xFF) / 255f - 0.485f) / 0.229f));
                imgData.put(idx + stride, (((pixelValue >> 8 & 0xFF) / 255f - 0.456f) / 0.224f));
                imgData.put(idx + stride * 2, (((pixelValue & 0xFF) / 255f - 0.406f) / 0.225f));
            }
        }

        imgData.rewind();
        return imgData;
    }

    // Preprocessing function for NudeNet 320n model (expects [batch, channels, height, width] with RGB format and [0,1] normalization)
    public static FloatBuffer preProcessForNudeNet(Bitmap bitmap) {
        // Ensure bitmap is exactly 320x320
        if (bitmap.getWidth() != IMAGE_SIZE_X || bitmap.getHeight() != IMAGE_SIZE_Y) {
            throw new IllegalArgumentException("Bitmap must be exactly " + IMAGE_SIZE_X + "x" + IMAGE_SIZE_Y);
        }
        
        FloatBuffer imgData = FloatBuffer.allocate(
                DIM_BATCH_SIZE
                        * DIM_PIXEL_SIZE
                        * IMAGE_SIZE_X
                        * IMAGE_SIZE_Y
        );
        imgData.rewind();
        int stride = IMAGE_SIZE_X * IMAGE_SIZE_Y;
        
        // Process pixels in chunks to reduce memory pressure
        int[] bmpData = new int[stride];
        bitmap.getPixels(bmpData, 0, IMAGE_SIZE_X, 0, 0, IMAGE_SIZE_X, IMAGE_SIZE_Y);
        
        // Process pixels more efficiently
        for (int i = 0; i < IMAGE_SIZE_X; i++) {
            for (int j = 0; j < IMAGE_SIZE_Y; j++) {
                int idx = IMAGE_SIZE_Y * i + j;
                int pixelValue = bmpData[idx];
                
                // Normalize to [0, 1] range for NudeNet model
                float r = (pixelValue >> 16 & 0xFF) / 255f;
                float g = (pixelValue >> 8 & 0xFF) / 255f;
                float b = (pixelValue & 0xFF) / 255f;
                
                // Store as [batch, channels, height, width] format
                imgData.put(idx, r);                    // R channel
                imgData.put(idx + stride, g);           // G channel  
                imgData.put(idx + stride * 2, b);       // B channel
            }
        }

        imgData.rewind();
        return imgData;
    }
}
