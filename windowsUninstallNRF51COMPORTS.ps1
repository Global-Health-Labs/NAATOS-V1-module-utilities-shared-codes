foreach ($dev in (Get-PnpDevice -class "Ports" | where {$_.Name -like "nRF52 SDFU USB*"})) {
  &"pnputil" /remove-device $dev.InstanceId 
}
Start-Sleep -Seconds 3