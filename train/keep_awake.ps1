# Run a .cmd file while keeping the laptop awake (no sleep or hibernate until it finishes).
#   powershell -File train\keep_awake.ps1 train\run_explain_all.cmd
param([Parameter(Mandatory = $true)][string]$Script)
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
try { & cmd.exe /c $Script } finally { [Win32.Power]::SetThreadExecutionState(0x80000000) | Out-Null }
