package org.vknrc.demo;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.os.Bundle;
import android.os.Build;

// NativeActivity loads its library before Activity.onCreate completes. Check it
// in a regular activity so loader errors can be displayed in a valid window.
public final class LauncherActivity extends Activity {
    @Override public void onCreate(Bundle savedState) {
        super.onCreate(savedState);
        try {
            System.loadLibrary("vknrc");
            startActivity(new Intent(this, VkNRCActivity.class));
            finish();
        } catch (UnsatisfiedLinkError error) {
            new AlertDialog.Builder(this).setTitle("VkNRC could not load")
                .setMessage("Device ABIs: " + String.join(", ", Build.SUPPORTED_ABIS) + "\n\n" + error.getMessage())
                .setPositiveButton("Close", (dialog, which) -> finish())
                .setOnCancelListener(dialog -> finish()).show();
        }
    }
}
