package org.vknrc.demo;

import android.app.Dialog;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.*;
import java.io.File;
import java.util.ArrayList;
import java.util.Locale;

final class RendererControls {
    // Shared with the render thread's AndroidControls protocol.
    private static final int LEFT=0, RIGHT=1, NEE=2, FAST=3, ROUGHNESS=4, INDIRECT=5,
        BOUNCES=6, RESOLUTION=7, ACCUMULATE=8, WINDOW=9, PAUSED=10, LOCK=11, EMA=12,
        SPP=13, MAX_WINDOW=14, READY=15, BENCHMARKING=16,
        RETRAIN=20, RESTART=21, BENCHMARK=22, CAPTURE=23, CANCEL=24, TRAIN_ONE=25;
    private static final int MATCH = ViewGroup.LayoutParams.MATCH_PARENT;
    private static final int WRAP = ViewGroup.LayoutParams.WRAP_CONTENT;
    private static final String[] METHODS = {"PT", "NRC", "Cache"};
    private static final String[] RESOLUTIONS = {"160 × 90", "320 × 180", "640 × 360", "960 × 540", "1280 × 720"};
    private interface Binding { void update(int[] state); }
    private final VkNRCActivity activity;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final ArrayList<Binding> bindings = new ArrayList<>();
    private int[] state;
    private String lastStatus = "";
    private File latestExr;
    private boolean capturePending;
    private int benchmarkSeconds = 5;
    private boolean includeHeader = true, resumed;
    private PopupWindow entry;
    private Dialog panel;
    private LinearLayout list;
    private TextView status;

    RendererControls(VkNRCActivity activity) { this.activity = activity; }

    void resume() { resumed = true; handler.removeCallbacks(poll); handler.post(poll); }
    void pause() {
        resumed = false;
        handler.removeCallbacks(poll);
        if (panel != null) panel.dismiss();
        if (entry != null) entry.dismiss();
    }

    private final Runnable poll = new Runnable() {
        @Override public void run() {
            if (!resumed || activity.isFinishing() || activity.isDestroyed()) return;
            state = VkNRCActivity.getRendererOptions();
            boolean ready = state != null && state.length > BENCHMARKING && state[READY] != 0;
            if (ready) {
                String message = VkNRCActivity.getRendererStatus();
                if (message != null && !message.equals(lastStatus)) {
                    lastStatus = message;
                    capturePending = false;
                    final String saved = "Saved linear EXR: ";
                    if (message.startsWith(saved)) latestExr = new File(message.substring(saved.length()));
                    if (!message.isEmpty()) Toast.makeText(activity, message, Toast.LENGTH_LONG).show();
                }
            }
            if (ready && panel != null) refreshPanel();
            if (ready && panel == null) showEntry();
            if (!ready) {
                if (panel != null) panel.dismiss();
                if (entry != null) entry.dismiss();
            }
            // The SPP readout is informational. Keep the overlay quiet between
            // updates so it remains accessible while the renderer is running.
            handler.postDelayed(this, 1500);
        }
    };

    private int dp(int size) { return Math.round(size * activity.getResources().getDisplayMetrics().density); }

    private void showEntry() {
        View decor = activity.getWindow().getDecorView();
        if (decor.getWindowToken() == null) return;
        if (entry == null) {
            Button open = new Button(activity);
            open.setText("Settings");
            open.setOnClickListener(v -> openPanel());
            // NativeActivity renders directly to the main surface. A separate
            // touchable window keeps this button above Vulkan and passes other touches through.
            entry = new PopupWindow(open, dp(104), dp(48), false);
            entry.setElevation(dp(4));
        }
        if (!entry.isShowing()) entry.showAtLocation(decor, Gravity.TOP | Gravity.END, dp(12), dp(8));
    }

    private TextView text(String value, int size) {
        TextView label = new TextView(activity);
        label.setText(value);
        label.setTextSize(size);
        return label;
    }
    private void section(String title) {
        TextView heading = text(title, 17);
        heading.setTypeface(null, Typeface.BOLD);
        heading.setPadding(0, dp(16), 0, dp(4));
        list.addView(heading);
    }
    private void send(int option, int value) {
        if (panel == null || state == null || state.length <= BENCHMARKING || state[READY] == 0) return;
        if (option < SPP) {
            if (state[option] == value) return;
            state[option] = value;
        }
        VkNRCActivity.setRendererOption(option, value);
        handler.removeCallbacks(poll);
        handler.postDelayed(poll, 120);
    }
    private static boolean editable(int[] s) { return s[PAUSED] == 0 && s[BENCHMARKING] == 0; }
    private void toggle(String label, int option) {
        Switch field = new Switch(activity);
        field.setText(label);
        field.setMinHeight(dp(46));
        list.addView(field);
        field.setOnCheckedChangeListener((button, on) -> send(option, on ? 1 : 0));
        bindings.add(s -> { field.setChecked(s[option] != 0); field.setEnabled(editable(s)); });
    }
    private void choice(String label, int option, String[] names) {
        LinearLayout row = new LinearLayout(activity);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.addView(text(label, 16), new LinearLayout.LayoutParams(0, WRAP, 1));
        Spinner field = new Spinner(activity);
        field.setContentDescription(label);
        ArrayAdapter<String> values = new ArrayAdapter<>(activity, android.R.layout.simple_spinner_item, names);
        values.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        field.setAdapter(values);
        field.setSelection(state[option], false);
        field.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            public void onItemSelected(AdapterView<?> parent, View view, int index, long id) { send(option, index); }
            public void onNothingSelected(AdapterView<?> parent) {}
        });
        row.addView(field, dp(172), dp(46));
        list.addView(row);
        bindings.add(s -> {
            if (field.getSelectedItemPosition() != s[option]) field.setSelection(s[option], false);
            field.setEnabled(editable(s));
        });
    }
    private static String rangeLabel(String title, int option, int value) {
        if (option == ROUGHNESS) return String.format(Locale.US, "%s: %.3f", title, value / 1000.0);
        if (option == WINDOW && value == 0) return title + ": all samples";
        return title + ": " + value;
    }
    private void range(String title, int option, int min, int max) {
        TextView label = text("", 16);
        SeekBar field = new SeekBar(activity);
        field.setContentDescription(title);
        list.addView(label);
        list.addView(field, MATCH, dp(38));
        field.setMax(max - min);
        field.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            private boolean tracking;
            private void apply(int value) {
                if (option == BENCHMARK) benchmarkSeconds = value;
                else send(option, value);
            }
            public void onStartTrackingTouch(SeekBar bar) { tracking = true; }
            public void onStopTrackingTouch(SeekBar bar) { tracking = false; apply(bar.getProgress() + min); }
            public void onProgressChanged(SeekBar bar, int progress, boolean user) {
                label.setText(rangeLabel(title, option, progress + min));
                if (user && !tracking) apply(progress + min);
            }
        });
        bindings.add(s -> {
            int value = option == BENCHMARK ? benchmarkSeconds : s[option];
            if (!field.isPressed()) {
                if (option == WINDOW) field.setMax(Math.min(max, Math.max(0, s[MAX_WINDOW])));
                field.setProgress(value - min);
                label.setText(rangeLabel(title, option, value));
            }
            boolean visible = option != ROUGHNESS || s[FAST] != 0;
            field.setVisibility(visible ? View.VISIBLE : View.GONE);
            label.setVisibility(field.getVisibility());
            field.setEnabled(editable(s));
        });
    }
    private void action(String title, int command) {
        Button button = new Button(activity);
        button.setText(title);
        button.setOnClickListener(v -> {
            if (command == CAPTURE) capturePending = true;
            int value = command == PAUSED ? 1 - state[PAUSED]
                : command == BENCHMARK ? benchmarkSeconds : command == CAPTURE && includeHeader ? 1 : 0;
            send(command, value);
            refreshPanel();
        });
        list.addView(button);
        bindings.add(s -> {
            boolean enabled = editable(s);
            if (command == PAUSED) {
                button.setText(s[PAUSED] == 0 ? "Pause rendering" : "Resume rendering");
                enabled = s[BENCHMARKING] == 0 && (s[PAUSED] != 0 || s[SPP] != 0);
            } else if (command == CAPTURE) {
                enabled = s[BENCHMARKING] == 0 && s[SPP] != 0;
            } else if (command == CANCEL || command == BENCHMARK) {
                boolean show = (command == CANCEL) == (s[BENCHMARKING] != 0);
                button.setVisibility(show ? View.VISIBLE : View.GONE);
                if (command == CANCEL) enabled = true;
            } else if (command == TRAIN_ONE) {
                button.setVisibility(s[LOCK] != 0 ? View.VISIBLE : View.GONE);
            }
            button.setEnabled(enabled);
        });
    }
    private void refreshPanel() {
        for (Binding binding : bindings) binding.update(state);
        status.setText(lastStatus);
        status.setVisibility(lastStatus.isEmpty() ? View.GONE : View.VISIBLE);
    }
    private void openPanel() {
        if (panel != null || state == null) return;
        if (!capturePending && (latestExr == null || !latestExr.isFile())) latestExr = activity.latestCapture();
        entry.dismiss();
        panel = new Dialog(activity);
        panel.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LinearLayout shell = new LinearLayout(activity);
        shell.setOrientation(LinearLayout.VERTICAL);
        shell.setPadding(dp(16), dp(8), dp(16), dp(8));
        shell.setBackgroundColor(Color.rgb(32, 34, 37));
        LinearLayout header = new LinearLayout(activity);
        header.setGravity(Gravity.CENTER_VERTICAL);
        TextView title = text("Render settings", 20);
        title.setTypeface(null, Typeface.BOLD);
        header.addView(title, new LinearLayout.LayoutParams(0, WRAP, 1));
        Button done = new Button(activity);
        done.setText("Done");
        done.setOnClickListener(v -> panel.dismiss());
        header.addView(done);
        shell.addView(header);
        TextView summary = text("", 14);
        shell.addView(summary);
        bindings.add(s -> summary.setText("Cornell Box  ·  " + Integer.toUnsignedString(s[SPP]) + " SPP"
            + (s[PAUSED] != 0 ? "  ·  Paused" : s[BENCHMARKING] != 0 ? "  ·  Benchmark running" : "")));
        ScrollView scroll = new ScrollView(activity);
        list = new LinearLayout(activity);
        list.setOrientation(LinearLayout.VERTICAL);
        scroll.addView(list);
        shell.addView(scroll, new LinearLayout.LayoutParams(MATCH, 0, 1));
        action("Pause rendering", PAUSED);
        section("View");
        choice("Left method", LEFT, METHODS);
        choice("Right method", RIGHT, METHODS);
        choice("Render resolution", RESOLUTION, RESOLUTIONS);
        section("Lighting");
        toggle("Next event estimation (NEE)", NEE);
        toggle("Indirect illumination only", INDIRECT);
        range("Max bounces", BOUNCES, 1, 64);
        toggle("Fast mode (NRC)", FAST);
        range("Min roughness", ROUGHNESS, 0, 1000);
        section("Accumulation");
        toggle("Accumulate samples", ACCUMULATE);
        range("Rolling window (SPP)", WINDOW, 0, 64);
        action("Restart accumulation", RESTART);
        list.addView(text("Start at zero SPP with the current camera and trained cache.", 14));
        section("Cache training");
        toggle("Lock training", LOCK);
        toggle("Use EMA weights", EMA);
        action("Train one frame", TRAIN_ONE);
        action("Retrain cache", RETRAIN);
        section("Capture");
        CheckBox metadata = new CheckBox(activity);
        metadata.setText("Include EXR header metadata");
        metadata.setChecked(includeHeader);
        metadata.setOnCheckedChangeListener((box, on) -> includeHeader = on);
        list.addView(metadata);
        action("Capture frame (EXR)", CAPTURE);
        Button saveExr = new Button(activity);
        saveExr.setText("Save EXR as…");
        saveExr.setOnClickListener(v -> activity.saveExrAs(latestExr));
        list.addView(saveExr);
        bindings.add(s -> saveExr.setEnabled(latestExr != null && s[BENCHMARKING] == 0));
        section("Benchmark");
        range("Benchmark duration (seconds)", BENCHMARK, 1, 120);
        action("Start benchmark", BENCHMARK);
        action("Cancel benchmark", CANCEL);
        status = text("", 14);
        status.setTextIsSelectable(true);
        list.addView(status);
        refreshPanel();
        panel.setContentView(shell);
        panel.setCanceledOnTouchOutside(true);
        panel.setOnDismissListener(dialog -> {
            panel = null;
            bindings.clear();
            if (resumed && state != null && state.length > BENCHMARKING && state[READY] != 0) showEntry();
        });
        panel.show();
        Window window = panel.getWindow();
        window.setGravity(Gravity.END | Gravity.CENTER_VERTICAL);
        window.setLayout(Math.min(dp(384), activity.getWindow().getDecorView().getWidth()), MATCH);
        WindowManager.LayoutParams params = window.getAttributes();
        params.dimAmount = 0.12f;
        window.setAttributes(params);
    }
}
