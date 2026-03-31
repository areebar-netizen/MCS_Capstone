import asyncio
from bleak import BleakClient

ADDRESS = "26D851CB-EE82-D07B-DF39-AC73337C4947"  # replace if different

async def run(address):
    print("Connecting to", address)
    try:
        async with BleakClient(address) as client:
            # is_connected is a boolean property, not a coroutine
            connected = client.is_connected
            print("Connected:", connected)
            if not connected:
                print("Client reports not connected after context manager entered.")
            # Some bleak/backends expose get_services coroutine, others populate a services property.
            svcs = None
            get_services_attr = getattr(client, 'get_services', None)
            if callable(get_services_attr):
                try:
                    svcs = await client.get_services()
                except Exception as e:
                    print("Failed to call get_services():", e)
            else:
                svcs = getattr(client, 'services', None)
            if svcs is None:
                print("No services available from this backend (cannot enumerate services).")
                return
            print("Services:")
            for s in svcs:
                print(s)
                for ch in s.characteristics:
                    print("  -", ch.uuid, ch.properties)
    except Exception as e:
        print("Connection failed:", e)

if __name__ == "__main__":
    asyncio.run(run(ADDRESS))