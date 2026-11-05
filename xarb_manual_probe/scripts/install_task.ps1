# Registers the recorder in Windows Task Scheduler for the current user: no admin rights, no open window.
# It starts at logon and every 5 minutes; while one copy runs, the next one leaves at once.
$root = Split-Path $PSScriptRoot -Parent
$python = (Resolve-Path "$root\..\..\..\.venv\Scripts\pythonw.exe").Path
$action = New-ScheduledTaskAction -Execute $python -Argument "scripts\run_x2.py record" -WorkingDirectory $root
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$every = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName "Fast_test_X2_recorder" -Action $action -Trigger $logon, $every -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "Fast_test_X2_recorder"
Get-ScheduledTask -TaskName "Fast_test_X2_recorder" | Select-Object TaskName, State
