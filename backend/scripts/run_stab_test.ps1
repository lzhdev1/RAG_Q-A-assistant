$env:Path = "C:\Program Files\Docker\Docker\resources\bin;$env:Path"
$logs = @()
foreach ($i in 1..4) {
    $out = docker run --rm -v "D:\AMySpace\AMe\RAG_Q-A-assistant\app:/app" -v "D:\AMySpace\AMe\RAG_Q-A-assistant\chroma_data:/app/chroma_data" -v "D:\AMySpace\AMe\RAG_Q-A-assistant\docs:/app/docs" -w /app --env-file "D:\AMySpace\AMe\RAG_Q-A-assistant\app\.env" rag_q-a-assistant-rag-app:latest python debug_query_methods.py 2>&1 | Out-String
    $lines = ($out -split "`n") | Where-Object { $_ -match "TARGET" }
    $logs += "--- Run$i : TARGET count=" + $lines.Count
    $logs += $lines
}
$logs | ForEach-Object { Write-Host $_ }
