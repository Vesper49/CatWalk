package com.example.medproject1;


import static android.content.ContentValues.TAG;

import androidx.appcompat.app.AppCompatActivity;


import android.graphics.Canvas;
import android.graphics.Color;

import android.graphics.Paint;
import android.os.Bundle;

import android.annotation.SuppressLint;

import android.graphics.Bitmap;
import android.media.Image;

import android.util.Log;
import android.util.Size;
import android.view.TextureView;
import android.widget.ImageView;

import androidx.annotation.NonNull;

import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import androidx.camera.lifecycle.ProcessCameraProvider;

import androidx.core.content.ContextCompat;

import com.google.android.gms.tasks.OnSuccessListener;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.firebase.ml.modeldownloader.CustomModel;
import com.google.firebase.ml.modeldownloader.CustomModelDownloadConditions;
import com.google.firebase.ml.modeldownloader.DownloadType;
import com.google.firebase.ml.modeldownloader.FirebaseModelDownloader;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.ExecutionException;

public class MainActivityTF extends AppCompatActivity{


    private ImageView preview;

    private TextureView textureView;

    Interpreter interpreter;

    ListenableFuture<ProcessCameraProvider> cameraProviderFuture;

    YUVtoRGB translator = new YUVtoRGB();

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_tf);
        preview = findViewById(R.id.imageP);
        CustomModelDownloadConditions conditions = new CustomModelDownloadConditions.Builder()
                .requireWifi()  // Also possible: .requireCharging() and .requireDeviceIdle()
                .build();
        FirebaseModelDownloader.getInstance()
                .getModel("Rat", DownloadType.LOCAL_MODEL_UPDATE_IN_BACKGROUND, conditions)
                .addOnSuccessListener(new OnSuccessListener<CustomModel>() {
                    @Override
                    public void onSuccess(CustomModel model) {

                        File modelFile = model.getFile();
                        if (modelFile != null) {
                            interpreter = new Interpreter(modelFile);
                        }
                    }
                });
        initializeCamera();
    }


    private void initializeCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {


                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                    ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                            .setTargetResolution(new Size(320, 320))
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build();

                    CameraSelector cameraSelector = new CameraSelector.Builder()
                            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                            .build();

                    imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(MainActivityTF.this),
                            new ImageAnalysis.Analyzer() {
                                @Override
                                public void analyze(@NonNull ImageProxy image) {
                                    @SuppressLint("UnsafeOptInUsageError") Image img = image.getImage();
                                    Bitmap bitmap = translator.translateYUV(img, MainActivityTF.this);
                                    Bitmap bitmap1 = Bitmap.createScaledBitmap(bitmap, 320, 320, false);
                                    ByteBuffer input = ByteBuffer.allocateDirect(320 * 320 * 3 * 4).order(ByteOrder.nativeOrder());
                                    for (int y = 0; y < 320; y++) {
                                        for (int x = 0; x < 320; x++) {
                                            int px = bitmap1.getPixel(x, y);

                                            // Get channel values from the pixel value.
                                            int r = Color.red(px);
                                            int g = Color.green(px);
                                            int b = Color.blue(px);


                                            float rf = (r - 127) / 255.0f;
                                            float gf = (g - 127) / 255.0f;
                                            float bf = (b - 127) / 255.0f;

                                            input.putFloat(rf);
                                            input.putFloat(gf);
                                            input.putFloat(bf);
                                        }
                                    }

                                    int bufferSize = 12 * java.lang.Float.SIZE / java.lang.Byte.SIZE;
                                    ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
                                    interpreter.run(input, modelOutput);

                                    modelOutput.rewind();
                                    FloatBuffer probabilities = modelOutput.asFloatBuffer();
                                    float score = probabilities.get(0);
                                    float location = probabilities.get(1);
                                    float num = probabilities.get(2);
                                    float categ = probabilities.get(3);
                                    Canvas canvas = new Canvas(bitmap1);
                                    int blue = Color.BLUE;
                                    Paint paint = new Paint();
                                    paint.setColor(blue);
                                    paint.setStyle(Paint.Style.STROKE);
                                    canvas.drawRect(score, location, num, categ, paint);



                                    preview.setRotation(image.getImageInfo().getRotationDegrees());
                                    preview.setImageBitmap(bitmap1);
                                    image.close();
                                }
                            });

                    cameraProvider.bindToLifecycle(MainActivityTF.this, cameraSelector, imageAnalysis);

                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

}