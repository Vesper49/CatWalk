package com.example.medproject1;

import static android.Manifest.permission.CAMERA;
import static android.Manifest.permission.RECORD_AUDIO;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.core.VideoCapture;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;
import java.io.File;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;



public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    StorageReference storageRef = FirebaseStorage.getInstance().getReference();
    StorageReference videoRef;
    PreviewView previewView;
    Uri videoUri;
    static String vidFilePath2;
    private VideoCapture videoCapture;
    private Button bRecord;
    private Button Menu;
    private Button TF;
    private static final int PERMISSION_REQUEST_CODE = 200;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Window w = getWindow();
        w.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.previewView);
        bRecord = findViewById(R.id.bRecord);
        bRecord.setText("Record"); // Set the initial text of the button

        bRecord.setOnClickListener(this);
        Menu = findViewById(R.id.Menu);
        Menu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openMenu();
            }
        });

        TF = findViewById(R.id.TF);
        TF.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openTF();
            }
        });

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                startCameraX(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, getExecutor());

        if(!checkPermission()){
            requestPermission();
        }

    }


    Executor getExecutor() {
        return ContextCompat.getMainExecutor(this);
    }

    @SuppressLint("RestrictedApi")
    private void startCameraX(ProcessCameraProvider cameraProvider) {
        cameraProvider.unbindAll();
        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();
        Preview preview = new Preview.Builder()
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        // Video capture use case
        videoCapture = new VideoCapture.Builder()
                .setVideoFrameRate(30)
                .build();

        //bind to lifecycle:
        cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, videoCapture);
    }


    @SuppressLint("RestrictedApi")
    @Override
    public void onClick(View view) {
        if (bRecord.getText() == "Record"){
            bRecord.setText("Stop");
            recordVideo();
        } else {
            bRecord.setText("Record");
            videoCapture.stopRecording();
            if(MenuMainActivity.getIfItDrive()){
                uploadVideoFirebase();
            }
        }
    }

    @SuppressLint("RestrictedApi")
    private void recordVideo() {
        if (videoCapture != null) {
            File movieDir = new File(Environment.getExternalStorageDirectory() + "/DCIM/CatWalk");
            if(!movieDir.exists()){
                movieDir.mkdir();
            }
            if(MenuMainActivity.getIfItDrive()){
                movieDir = new File(Environment.getExternalStorageDirectory() + "/DCIM/CatWalk/Firebase");
            }
            else {
                movieDir = new File(Environment.getExternalStorageDirectory() + "/DCIM/CatWalk/YandexDisk");
            }
            if(!movieDir.exists()){
                movieDir.mkdir();
            }
            int count = 1;
            String vidFilePath = movieDir.getAbsolutePath() + "/" + "Video" + count + ".mp4";
            File vidFile = new File(vidFilePath);;
            if(vidFile.exists()){
                while(vidFile.exists()){
                    vidFilePath = movieDir.getAbsolutePath() + "/" + "Video" + count + ".mp4";
                    vidFile = new File(vidFilePath);
                    ++count;
                }
            }
            vidFilePath2 = vidFilePath;
            try {
                if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    // TODO: Consider calling
                    //    ActivityCompat#requestPermissions
                    // here to request the missing permissions, and then overriding
                    //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                    //                                          int[] grantResults)
                    // to handle the case where the user grants the permission. See the documentation
                    // for ActivityCompat#requestPermissions for more details.
                    return;
                }
                videoCapture.startRecording(
                        new VideoCapture.OutputFileOptions.Builder(vidFile).build(),
                        getExecutor(),
                        new VideoCapture.OnVideoSavedCallback() {
                            @Override
                            public void onVideoSaved(@NonNull VideoCapture.OutputFileResults outputFileResults) {
                                Toast.makeText(MainActivity.this, "Video has been saved successfully.", Toast.LENGTH_SHORT).show();
                            }

                            @Override
                            public void onError(int videoCaptureError, @NonNull String message, @Nullable Throwable cause) {
                                Toast.makeText(MainActivity.this, "Error saving video: " + message, Toast.LENGTH_SHORT).show();
                            }
                        }
                );
            } catch (Exception e) {
                e.printStackTrace();
            }

        }

    }

    private void uploadVideoFirebase(){
        videoUri = Uri.fromFile(new File(vidFilePath2));
        videoRef = storageRef.child("Videos/" + videoUri.getLastPathSegment());
        UploadTask uploadTask = videoRef.putFile(videoUri);
        uploadTask.addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                Toast.makeText(MainActivity.this, "Error upload video ", Toast.LENGTH_SHORT).show();
            }
        }).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
            @Override
            public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                Toast.makeText(MainActivity.this, "Uploaded", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private boolean checkPermission() {
        int result = ContextCompat.checkSelfPermission(getApplicationContext(), RECORD_AUDIO);
        int result1 = ContextCompat.checkSelfPermission(getApplicationContext(), CAMERA);

        return result == PackageManager.PERMISSION_GRANTED && result1 == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermission() {

        ActivityCompat.requestPermissions(this, new String[]{CAMERA, RECORD_AUDIO}, PERMISSION_REQUEST_CODE);

    }

    public void openMenu(){
        Intent intent = new Intent(this, MenuMainActivity.class);
        startActivity(intent);
    }

    public void openTF(){
        Intent intent = new Intent(this, MainActivityTF.class);
        startActivity(intent);
    }

    public void uploadVideoYD(){

    }

}
