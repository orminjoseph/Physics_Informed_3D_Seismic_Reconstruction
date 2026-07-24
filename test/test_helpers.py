from utils.helpers import (
    set_seed,
    get_device,
    create_directory
)

print("=" * 50)
print("Testing Helper Functions")
print("=" * 50)

set_seed(42)

device = get_device()

print("Device:", device)

create_directory("sample_results")

print("Directory created successfully.")

print("=" * 50)