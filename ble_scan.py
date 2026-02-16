import asyncio
import sys
from bleak import BleakScanner

async def scan(timeout=8.0):
    print("Scanning for BLE devices for", timeout, "seconds...")
    try:
        devices = await BleakScanner.discover(timeout=timeout)
    except Exception as e:
        print("Error while scanning:", e)
        print("On macOS you may need to grant Bluetooth permissions to Terminal/IDE in System Settings -> Privacy & Security -> Bluetooth.")
        return
    if not devices:
        print("No devices found.")
        return
    for d in devices:
        # d.metadata may contain manufacturer data on some platforms; be defensive
        meta = getattr(d, "metadata", {}) or {}
        name = d.name or meta.get('local_name') or meta.get('name')
        # RSSI attribute varies across bleak/backends; access defensively
        rssi = getattr(d, 'rssi', None)
        if rssi is None:
            rssi = meta.get('rssi') if isinstance(meta, dict) else None
        if rssi is None:
            details = getattr(d, 'details', None) or {}
            rssi = details.get('rssi') if isinstance(details, dict) else None
        rssi_str = f"RSSI={rssi}" if rssi is not None else "RSSI=N/A"
        print(f"{d.address}  |  {name}  |  {rssi_str}")

if __name__ == "__main__":
    timeout = 8.0
    if len(sys.argv) > 1:
        try:
            timeout = float(sys.argv[1])
        except ValueError:
            pass
    asyncio.run(scan(timeout))
