$env:Path = "C:\Program Files\Docker\Docker\resources\bin;$env:Path"
docker run --rm -v "D:\AMySpace\AMe\RAG_Q-A-assistant\app:/app" -w /app rag_q-a-assistant-rag-app:latest python debug_sync_threshold.py *> "$env:TEMP\sync_test.txt"
Write-Host "exit=$LASTEXITCODE"
Get-Content -LiteralPath "$env:TEMP\sync_test.txt" -Encoding utf8 | Select-Object -Last 25
