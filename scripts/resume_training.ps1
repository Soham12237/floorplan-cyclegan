# Resume the floorplan CycleGAN run from checkpoints\floorplan_cyclegan\latest_*.pth.
#
# Original run: 25 epochs at constant LR + 25 epochs of linear decay (50 total).
# `latest` was written at epoch 14, iteration 250, so we restart the count at epoch 14
# (the first 250 iterations of that epoch are redone; the LR schedule stays correct).
# Optimizer state is not saved by this repo, so Adam restarts fresh - that is normal.
#
# Close games / other GPU-heavy apps first: training needs ~3.9 GB of the 4 GB VRAM.
# Usage:  powershell -File scripts\resume_training.ps1
param([int]$EpochCount = 14)

$root = Split-Path -Parent $PSScriptRoot
$env:TMP = "$root\.tmp"; $env:TEMP = $env:TMP
$env:PYTHONUNBUFFERED = "1"   # so logs\train.log fills in live

$argList = "train.py --dataroot ..\data\cyclegan --name floorplan_cyclegan --checkpoints_dir ..\checkpoints " +
           "--model cycle_gan --load_size 256 --crop_size 256 --preprocess resize_and_crop " +
           "--n_epochs 25 --n_epochs_decay 25 --save_epoch_freq 5 --num_threads 2 " +
           "--continue_train --epoch latest --epoch_count $EpochCount"

$p = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList $argList `
     -WorkingDirectory "$root\cyclegan" `
     -RedirectStandardOutput "$root\logs\train_resume.log" -RedirectStandardError "$root\logs\train_resume.err.log" `
     -WindowStyle Hidden -PassThru
$p.Id | Out-File "$root\logs\train.pid" -Encoding ascii
"resumed from epoch $EpochCount, launcher PID $($p.Id) at $(Get-Date -Format HH:mm:ss)"
