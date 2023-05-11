package com.example.medproject1;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class MenuMainActivity extends AppCompatActivity {

    private Button back;
    private Button changeCloudButton;
    TextView cloudText;
    public static boolean ifItDrive = false; // if false it means Drive, if not Firebase
    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.menu_main);
        back = findViewById(R.id.back);
        back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                doBack();
            }
        });
        changeCloudButton = findViewById(R.id.Change);
        changeCloudButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                changeCloud();
            }
        });
        cloudText = findViewById(R.id.Cloud);
        cloudText.setText("YandexDisk");

    }

    public void doBack(){
        Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);
    }

    public void changeCloud(){
        if(cloudText.getText() == "YandexDisk"){
            cloudText.setText("Firebase");
            ifItDrive = true;
        }
        else {
            cloudText.setText("YandexDisk");
            ifItDrive = false;
        }
    }

    public static boolean getIfItDrive() {
        return ifItDrive;
    }
}