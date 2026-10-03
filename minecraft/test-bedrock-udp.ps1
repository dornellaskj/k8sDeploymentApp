param(
    [string]$ServerIp = "192.168.86.164",
    [int]$Port = 19132,
    [int]$TimeoutMs = 3000
)

$magic = [byte[]](0x00,0xFF,0xFF,0x00,0xFE,0xFE,0xFE,0xFE,0xFD,0xFD,0xFD,0xFD,0x12,0x34,0x56,0x78)
$timestamp = [BitConverter]::GetBytes([int64]0)
$clientGuid = [BitConverter]::GetBytes([int64]12345)
$packet = [byte[]](0x01) + $timestamp + $magic + $clientGuid

$udp = New-Object System.Net.Sockets.UdpClient
$udp.Client.ReceiveTimeout = $TimeoutMs
$endpoint = New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Parse($ServerIp), $Port)

try {
    $udp.Send($packet, $packet.Length, $endpoint) | Out-Null
    $remote = New-Object System.Net.IPEndPoint ([System.Net.IPAddress]::Any, 0)
    $response = $udp.Receive([ref]$remote)
    Write-Host "Reply received from $($remote.Address):$($remote.Port) - $($response.Length) bytes" -ForegroundColor Green
    Write-Host "Server is reachable." -ForegroundColor Green
} catch [System.Net.Sockets.SocketException] {
    Write-Host "No reply within $TimeoutMs ms - server unreachable or UDP blocked." -ForegroundColor Red
    Write-Host $_.Exception.Message
} finally {
    $udp.Close()
}
