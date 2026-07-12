# Push in chunks to avoid GitHub's HTTP 500 on large pushes
# This script splits the single commit into multiple smaller commits

$ErrorActionPreference = "Stop"

# Save current branch
$currentBranch = "main"
$tempBranch = "temp-chunked-push"

Write-Host "=== Step 1: Create orphan branch ===" -ForegroundColor Cyan
git checkout --orphan $tempBranch
git reset

# Batch 1: Code files only (small - ~5MB)
Write-Host "`n=== Batch 1: Code files + README ===" -ForegroundColor Green
git add .gitignore README.md requirements.txt *.py *.csv src/
git commit -m "Add code files and source"
git push origin "${tempBranch}:main" --force
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Batch 1" -ForegroundColor Red; exit 1 }
Write-Host "Batch 1 pushed successfully!" -ForegroundColor Green

# Batch 2: data/embeddings + data/dev (~260MB)
Write-Host "`n=== Batch 2: data/embeddings + data/dev ===" -ForegroundColor Green
git add data/embeddings/ data/dev/
git commit -m "Add embeddings and dev data"
git push origin "${tempBranch}:main"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Batch 2" -ForegroundColor Red; exit 1 }
Write-Host "Batch 2 pushed successfully!" -ForegroundColor Green

# Batch 3: data/image/chinese + data/image/malayalam + data/image/tamil (~391MB)
Write-Host "`n=== Batch 3: data/image (chinese, malayalam, tamil) ===" -ForegroundColor Green
git add data/image/chinese/ data/image/malayalam/ data/image/tamil/
git commit -m "Add chinese, malayalam, tamil images"
git push origin "${tempBranch}:main"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Batch 3" -ForegroundColor Red; exit 1 }
Write-Host "Batch 3 pushed successfully!" -ForegroundColor Green

# Batch 4: First half of data/image/western (files 1-4700, ~780MB)
Write-Host "`n=== Batch 4: data/image/western (first half) ===" -ForegroundColor Green
$westernFiles = Get-ChildItem -File e:\pep\data\image\western | Sort-Object Name
$half = [math]::Floor($westernFiles.Count / 2)
$firstHalf = $westernFiles[0..($half-1)]
foreach ($f in $firstHalf) {
    git add "data/image/western/$($f.Name)"
}
git commit -m "Add western images (part 1)"
git push origin "${tempBranch}:main"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Batch 4" -ForegroundColor Red; exit 1 }
Write-Host "Batch 4 pushed successfully!" -ForegroundColor Green

# Batch 5: Second half of data/image/western + remaining CSV
Write-Host "`n=== Batch 5: data/image/western (second half) ===" -ForegroundColor Green
$secondHalf = $westernFiles[$half..($westernFiles.Count-1)]
foreach ($f in $secondHalf) {
    git add "data/image/western/$($f.Name)"
}
# Add any remaining files in data/image/western (like CSV)
git add data/image/western/
git commit -m "Add western images (part 2)"
git push origin "${tempBranch}:main"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Batch 5" -ForegroundColor Red; exit 1 }
Write-Host "Batch 5 pushed successfully!" -ForegroundColor Green

# Batch 6: dev/ train/ test/ results/
Write-Host "`n=== Batch 6: dev, train, test, results ===" -ForegroundColor Green
git add dev/ train/ test/ results/
git commit -m "Add dev, train, test, and results"
git push origin "${tempBranch}:main"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Batch 6" -ForegroundColor Red; exit 1 }
Write-Host "Batch 6 pushed successfully!" -ForegroundColor Green

# Batch 7: Anything remaining
Write-Host "`n=== Batch 7: Any remaining files ===" -ForegroundColor Green
git add -A
$status = git status --porcelain
if ($status) {
    git commit -m "Add remaining files"
    git push origin "${tempBranch}:main"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED at Batch 7" -ForegroundColor Red; exit 1 }
    Write-Host "Batch 7 pushed successfully!" -ForegroundColor Green
} else {
    Write-Host "Nothing remaining to add." -ForegroundColor Yellow
}

# Switch back to main and clean up
Write-Host "`n=== Cleanup: Reset main to match remote ===" -ForegroundColor Cyan
git checkout main
git branch -D $tempBranch
git pull origin main --rebase

Write-Host "`n=== ALL DONE! Repository pushed successfully! ===" -ForegroundColor Green
