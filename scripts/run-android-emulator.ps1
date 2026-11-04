# Launch the local test AVD, install the x86_64 build and show VkNRC.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$toolsDir = Join-Path $repo 'build/android-tools'
$apk = Join-Path $repo 'build/android/VkNRC-x86_64-debug.apk'
if (!(Test-Path $apk)) { throw "Missing $apk. Build the x86_64 APK first (see docs/android-build.md)." }
if (Test-Path "$toolsDir/sdk/emulator/emulator.exe") {
    $env:ANDROID_HOME = "$toolsDir/sdk"
    $env:ANDROID_USER_HOME = "$toolsDir/user"
    $env:ANDROID_EMULATOR_HOME = "$toolsDir/user"
    $env:ANDROID_AVD_HOME = "$toolsDir/avd"
}
$adb = Join-Path $env:ANDROID_HOME 'platform-tools/adb.exe'
$emulator = Join-Path $env:ANDROID_HOME 'emulator/emulator.exe'
$avd = 'VkNRC_API35'
$serial = 'emulator-5572'
if (!((& $emulator -list-avds) -contains $avd)) { throw "Create the $avd virtual device first (see docs/android-build.md)." }
$devices = & $adb devices
if ($LASTEXITCODE) { throw 'Could not connect to the local adb server.' }
if ($devices -match "^$serial\s") {
    $runningAVD = & $adb -s $serial emu avd name
    if (!($runningAVD -contains $avd)) { throw "Port 5572 is in use by another emulator." }
} else {
    # This window is the interactive virtual phone used to test the demo.
    $process = Start-Process $emulator -ArgumentList '-avd',$avd,'-port','5572','-gpu','host','-no-snapshot' -PassThru
}
$deadline = (Get-Date).AddMinutes(3)
do {
    if ($process -and $process.HasExited) { throw "Emulator exited early. Check 'emulator -accel-check' and the AVD log." }
    if (((& $adb devices) -match "^$serial\s+device$") -and ((& $adb -s $serial shell getprop sys.boot_completed) -eq '1')) { break }
    if ((Get-Date) -gt $deadline) { throw "Timed out waiting for $avd to boot." }
    Start-Sleep -Seconds 2
} while ($true)
& $adb -s $serial install -r $apk
if ($LASTEXITCODE) { throw "APK installation failed." }
& $adb -s $serial shell wm dismiss-keyguard
& $adb -s $serial shell am start -W -n org.vknrc.demo/.LauncherActivity
if ($LASTEXITCODE) { throw "VkNRC launch failed." }
