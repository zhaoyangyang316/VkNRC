package org.vknrc.demo;

import android.app.AlertDialog;
import android.app.NativeActivity;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.DocumentsContract;
import android.util.Log;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;

public final class VkNRCActivity extends NativeActivity {
    private static native int[] renderCornellPreview(String objPath);
    static native int[] getRendererOptions();
    static native void setRendererOption(int option, int value);
    static native String getRendererStatus();
    private RendererControls controls;
    private static final int SAVE_EXR = 101;
    private static final String EXR_SOURCE = "pendingExrSource";
    private File pendingExr;

    @Override protected void onCreate(Bundle savedState) {
        System.loadLibrary("vknrc");
        super.onCreate(savedState);
        controls = new RendererControls(this);
        if (savedState != null && savedState.getString(EXR_SOURCE) != null)
            pendingExr = new File(savedState.getString(EXR_SOURCE));
    }

    @Override protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        if (pendingExr != null) out.putString(EXR_SOURCE, pendingExr.getPath());
    }

    File latestCapture() {
        File newest = null;
        for (File root : new File[]{getExternalFilesDir(null), getFilesDir()}) {
            if (root == null) continue;
            File[] files = new File(root, "captures").listFiles((dir, name) ->
                name.startsWith("frame-") && name.endsWith(".exr"));
            if (files == null) continue;
            for (File file : files)
                if (file.isFile() && file.length() > 0 &&
                    (newest == null || file.lastModified() > newest.lastModified())) newest = file;
        }
        return newest;
    }

    void saveExrAs(File source) {
        if (source == null || !source.isFile()) {
            Toast.makeText(this, "Capture a frame before saving an EXR.", Toast.LENGTH_LONG).show();
            return;
        }
        Intent create = new Intent(Intent.ACTION_CREATE_DOCUMENT)
            .addCategory(Intent.CATEGORY_OPENABLE).setType("image/x-exr")
            .putExtra(Intent.EXTRA_TITLE, source.getName())
            .putExtra(DocumentsContract.EXTRA_INITIAL_URI, DocumentsContract.buildDocumentUri(
                "com.android.externalstorage.documents", "primary:" + Environment.DIRECTORY_DOWNLOADS));
        pendingExr = source;
        try {
            startActivityForResult(create, SAVE_EXR);
        } catch (ActivityNotFoundException | SecurityException error) {
            pendingExr = null;
            Toast.makeText(this, "Cannot open the document picker: " + error.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != SAVE_EXR) return;
        File source = pendingExr;
        pendingExr = null;
        if (result != RESULT_OK) return;
        Uri target = data == null ? null : data.getData();
        if (source == null || target == null) {
            Toast.makeText(this, "Could not recover the capture or selected document.", Toast.LENGTH_LONG).show();
            return;
        }
        Context app = getApplicationContext();
        new Thread(() -> copyExr(app, source, target), "VkNRC-EXR-export").start();
    }

    private static void copyExr(Context app, File source, Uri target) {
        String message;
        try (FileInputStream input = new FileInputStream(source)) {
            long size = input.getChannel().size(), copied = 0;
            if (size == 0) throw new IOException("The source capture is empty.");
            try (OutputStream output = app.getContentResolver().openOutputStream(target, "wt")) {
                if (output == null) throw new IOException("Document provider is unavailable.");
                byte[] buffer = new byte[65536];
                int count;
                while ((count = input.read(buffer)) != -1) {
                    output.write(buffer, 0, count);
                    copied += count;
                }
                if (copied != size) throw new IOException("The source capture changed during export.");
            }
            message = "EXR saved to the selected document.";
            Log.i("VkNRC", "Exported " + source.getName() + " (" + copied + " bytes)");
        } catch (IOException | RuntimeException error) {
            message = "Could not save EXR: " + error.getMessage();
            Log.w("VkNRC", message, error);
        }
        final String result = message;
        new Handler(Looper.getMainLooper()).post(() -> Toast.makeText(app, result, Toast.LENGTH_LONG).show());
    }

    @Override protected void onResume() {
        super.onResume();
        controls.resume();
    }

    @Override protected void onPause() {
        controls.pause();
        super.onPause();
    }

    // Called by the native render thread when the device cannot run the renderer.
    public void showRendererError(String message) {
        runOnUiThread(() -> {
            if (isFinishing() || isDestroyed()) return;
            if (controls != null) controls.pause();
            LinearLayout content = new LinearLayout(this);
            content.setOrientation(LinearLayout.VERTICAL);
            int padding = (int) (16 * getResources().getDisplayMetrics().density);
            content.setPadding(padding, padding, padding, padding);
            TextView text = new TextView(this);
            text.setText(message);
            text.setTextIsSelectable(true);
            content.addView(text);
            ScrollView scroll = new ScrollView(this);
            scroll.addView(content);
            AlertDialog report = new AlertDialog.Builder(this).setTitle("VkNRC renderer")
                .setView(scroll).setPositiveButton("Close", (dialog, which) -> finish())
                .setOnCancelListener(dialog -> finish()).show();

            String scene = new File(getFilesDir(), "cornell-box/cornell-box.obj").getPath();
            new Thread(() -> {
                try {
                    // Also register the JNI methods if the process was recreated directly into this activity.
                    System.loadLibrary("vknrc");
                    int[] pixels = renderCornellPreview(scene);
                    if (pixels == null || pixels.length != 160 * 90) return;
                    Bitmap bitmap = Bitmap.createBitmap(pixels, 160, 90, Bitmap.Config.ARGB_8888);
                    runOnUiThread(() -> {
                        if (!report.isShowing() || isDestroyed()) return;
                        TextView label = new TextView(this);
                        label.setText("CPU preview • NRC unavailable");
                        content.addView(label, 0);
                        ImageView image = new ImageView(this);
                        image.setImageBitmap(bitmap);
                        image.setAdjustViewBounds(true);
                        image.setMaxHeight(padding * 12);
                        image.setContentDescription("Cornell Box CPU reference render");
                        content.addView(image, 1);
                    });
                } catch (RuntimeException | LinkageError error) {
                    Log.w("VkNRC", "CPU preview unavailable", error);
                }
            }, "VkNRC-CPU-preview").start();
        });
    }
}
