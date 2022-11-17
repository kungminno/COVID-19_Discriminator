package com.example.capstone;

import static android.content.ContentValues.TAG;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.widget.TextViewCompat;

import com.bumptech.glide.Glide;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.sql.Struct;
import java.util.concurrent.atomic.AtomicBoolean;

@RequiresApi(api = Build.VERSION_CODES.CUPCAKE)
public class RecordActivity extends AppCompatActivity {
    public SendData mSendData;
    public PlayData mPlayData;
    public ResetHandler resetHandler = new ResetHandler();
    public LoadHandler loadHandler = new LoadHandler();

    private static final String IP = "117.16.123.50";
    private static final int PORT = 9999;
    private Socket socket;
    private DataOutputStream dos;
    private DataInputStream dis;
    private String input_message;

    private Button startButton;
    private Button playButton;
    private TextView txtView;
    private ImageView imageView;

    private AudioRecord audioRecord = null;

    public static final String recordPermission = Manifest.permission.RECORD_AUDIO;
    public static final int PERMISSION_CODE = 21;

    private static final int SAMPLING_RATE_IN_HZ = 48000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_STEREO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int BUFFER_BYTE_SIZE = SAMPLING_RATE_IN_HZ * 20;

    private final byte[] BufferTrack = new byte[SAMPLING_RATE_IN_HZ];
    private final byte[] BufferRecord = new byte[SAMPLING_RATE_IN_HZ];
    private final ByteBuffer byteBuffer = ByteBuffer.allocate(BUFFER_BYTE_SIZE);

    private final AtomicBoolean recordingInProgress = new AtomicBoolean(false);

    private long backKeyPressedTime = 0;


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_record);

        txtView = findViewById(R.id.textView);
        TextViewCompat.setAutoSizeTextTypeWithDefaults(txtView, TextViewCompat.AUTO_SIZE_TEXT_TYPE_UNIFORM);
        startButton = findViewById(R.id.btnStartRec);
        playButton = findViewById(R.id.btnStartPlay);
        imageView = findViewById(R.id.imageView);
        Glide.with(this).load(R.raw.recording2).into(imageView);

        startButton.setEnabled(false);
        playButton.setEnabled(false);

        mSendData = new SendData();
        mSendData.start();

        startButton.setOnClickListener(v -> {
            if (checkNetworkState(RecordActivity.this)) {
                // 녹음 버튼을 누르면 바로 녹음중 화면으로 변경
                Message reset_message = resetHandler.obtainMessage();
                resetHandler.sendMessage(reset_message);

                startButton.setEnabled(false);
                playButton.setEnabled(false);

                mSendData = new SendData();
                mSendData.start();
            }
        });

        playButton.setOnClickListener(v -> {
            startButton.setEnabled(false);
            playButton.setEnabled(false);

            mPlayData = new PlayData();
            mPlayData.start();
        });
    }

    @Override
    public void onBackPressed() {
        if (System.currentTimeMillis() > backKeyPressedTime + 2000) {
            backKeyPressedTime = System.currentTimeMillis();
            Toast.makeText(this, "종료하려면 뒤로 가기를 한 번 더 누르세요.", Toast.LENGTH_SHORT).show();
            return;
        }

        if (System.currentTimeMillis() <= backKeyPressedTime + 2000) {
            finish();
        }
    }

    public boolean checkNetworkState(Context context) {
        ConnectivityManager manager = (ConnectivityManager) context.getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo wifi = manager.getNetworkInfo(ConnectivityManager.TYPE_WIFI);
        NetworkInfo mobile = manager.getNetworkInfo(ConnectivityManager.TYPE_MOBILE);

        if (wifi != null && wifi.isConnected()) {
            return true;
        } else if (mobile != null && mobile.isConnected()) {
            return true;
        } else {
            Toast.makeText(this, "네트워크 연결이 필요합니다.", Toast.LENGTH_SHORT).show();
        }

        return false;
    }

    class SendData extends Thread{
        public void run(){
            if (ActivityCompat.checkSelfPermission(getApplicationContext(), recordPermission) == PackageManager.PERMISSION_GRANTED) {
                audioRecord = new AudioRecord(MediaRecorder.AudioSource.DEFAULT,
                        SAMPLING_RATE_IN_HZ,
                        CHANNEL_CONFIG,
                        AUDIO_FORMAT,
                        SAMPLING_RATE_IN_HZ);

                if (AudioRecord.STATE_INITIALIZED == audioRecord.getState()) {
                    recordingInProgress.set(true);

                    audioRecord.startRecording();

                    byteBuffer.rewind();

                    int retBufferSize;

                    try {
                        //Log.w("client", "서버 연결 시도");
                        socket = new Socket();
                        socket.connect(new InetSocketAddress(IP, PORT), 1000);
                        //Log.w("client", "서버 접속 성공");
                    } catch (IOException e) {
                        Log.w("error", e);
                        e.printStackTrace();
                    }
                    //Log.w("client", "안드로이드에서 서버로 연결요청");

                    try {
                        dos = new DataOutputStream(socket.getOutputStream());
                        dis = new DataInputStream(socket.getInputStream());
                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.w("error", e);
                    }
                    //Log.w("client", "버퍼 생성 성공");

                    try {
                        Message hold_message = loadHandler.obtainMessage();
                        loadHandler.sendMessage(hold_message);

                        while ((byteBuffer.position() + SAMPLING_RATE_IN_HZ) < BUFFER_BYTE_SIZE) {
                            retBufferSize = audioRecord.read(BufferRecord, 0, SAMPLING_RATE_IN_HZ);
                            byteBuffer.put(BufferRecord, 0, retBufferSize);

                            dos.write(BufferRecord);
                            dos.flush();
                        }
                        //Log.w("client", "전송 완료");

                        byte[] buf = new byte[512];
                        //Log.w("client", "수신 대기 중");
                        int read_Byte  = dis.read(buf);
                        input_message = new String(buf, 0, read_Byte);
                        //Log.w("client", "수신 완료");

                        socket.close();
                    } catch (Exception e) {
                        Log.w("error", e);
                    }

                    audioRecord.stop();
                    audioRecord.release();

                    input_message = "positive 90";
                    Message msg = handler.obtainMessage();
                    handler.sendMessage(msg);
                }
            } else {
                ActivityCompat.requestPermissions(RecordActivity.this, new String[]{recordPermission}, PERMISSION_CODE);
            }
        }

        @SuppressLint("HandlerLeak")
        final Handler handler = new Handler() {
            @SuppressLint("SetTextI18n")
            public void handleMessage(@NonNull Message msg) {
                if(input_message != null) {
                    String[] msgArr = input_message.split("\\s");

                    if (msgArr[0].equalsIgnoreCase("POSITIVE")) {
                        txtView.setTextColor(Color.parseColor("#EE334E"));
                        txtView.setText("코로나19 기침소리 판별 결과 양성(Positive) 입니다.\n가까운 병원에서 검사를 받아보세요.\n\n예측률 : " + msgArr[1] + "%");
                        imageView.setImageResource(R.drawable.warning);
                    } else if (msgArr[0].equalsIgnoreCase("NEGATIVE")) {
                        txtView.setTextColor(Color.parseColor("#00A2E5"));
                        txtView.setText("코로나19 기침소리 판별 결과 음성(Negative) 입니다.\n\n예측률 : " + msgArr[1] + "%");
                        imageView.setImageResource(R.drawable.safe);
                    } else if (msgArr[0].equalsIgnoreCase("RETRY")) {
                        txtView.setTextColor(Color.parseColor("#857C7A"));
                        txtView.setText("다시 시도해 주세요.");
                        imageView.setImageResource(R.drawable.retry);
                    }

                    startButton.setEnabled(true);
                    playButton.setEnabled(true);
                }
            }
        };
    }

    class PlayData extends Thread{
        public void run() {
            if (null == audioRecord) {
                Toast.makeText(RecordActivity.this, "녹음된 파일이 없습니다.\n먼저 녹음해 주세요.", Toast.LENGTH_SHORT).show();
            } else {
                try {
                    AudioTrack audioTrack = new AudioTrack(AudioManager.STREAM_MUSIC,
                            SAMPLING_RATE_IN_HZ,
                            CHANNEL_CONFIG,
                            AUDIO_FORMAT,
                            SAMPLING_RATE_IN_HZ,
                            AudioTrack.MODE_STREAM);

                    byteBuffer.position(0);

                    while(byteBuffer.position() <= (BUFFER_BYTE_SIZE - SAMPLING_RATE_IN_HZ)) {
                        byteBuffer.get(BufferTrack, 0, SAMPLING_RATE_IN_HZ);
                        audioTrack.write(BufferTrack, 0, SAMPLING_RATE_IN_HZ);
                        audioTrack.play();
                    }

                    audioTrack.stop();
                    audioTrack.release();

                    Message msg = handler.obtainMessage();
                    handler.sendMessage(msg);
                } catch (Exception e){
                    Log.e(TAG, "Exception: " + e);
                }
            }
        }
        @SuppressLint("HandlerLeak")
        final Handler handler = new Handler() {
            public void handleMessage(@NonNull Message msg) {
                startButton.setEnabled(true);
            }
        };
    }

    class ResetHandler extends Handler {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);

            Glide.with(RecordActivity.this).load(R.raw.recording2).into(imageView);
            txtView.setTextColor(Color.parseColor("#857C7A"));
            txtView.setText("녹음 중...");
        }
    }

    class LoadHandler extends Handler {
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);

            Glide.with(RecordActivity.this).load(R.raw.loading).into(imageView);
            txtView.setTextColor(Color.parseColor("#857C7A"));
            txtView.setText("진단 중...");
        }
    }
}