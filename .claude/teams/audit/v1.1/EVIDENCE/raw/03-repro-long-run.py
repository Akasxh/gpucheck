"""Long-run cold-start of 1024^3 fp32 — does it ever transition?"""
import time
import torch

def event(fn):
    torch.mps.synchronize(); t = time.perf_counter()
    fn(); torch.mps.synchronize()
    return (time.perf_counter() - t) * 1000

assert torch.backends.mps.is_available()

print("=== Test A: 100 cold-start calls of 1024^3 fp32 ===")
a = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
b = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
for i in range(100):
    ms = event(lambda: a @ b)
    if i < 5 or i % 10 == 0:
        print(f"  call {i:3d}: {ms:.4f} ms")

print("\n=== Test B: Cold-start 1024^3 fp32 with NEW tensors each call ===")
for i in range(20):
    a = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
    b = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
    ms = event(lambda: a @ b)
    if i < 5 or i % 5 == 0:
        print(f"  call {i:3d}: {ms:.4f} ms")

print("\n=== Test C: Cold then sequence: 1024 fp32, then 1023^3 fp32 (kernel-cache miss?), then 1024 fp32 again ===")
a = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
b = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
print(f"  1024 cold: {event(lambda: a @ b):.4f} ms")
print(f"  1024 cold: {event(lambda: a @ b):.4f} ms")
a2 = torch.randn(1023, 1023, device='mps', dtype=torch.float32)
b2 = torch.randn(1023, 1023, device='mps', dtype=torch.float32)
print(f"  1023:      {event(lambda: a2 @ b2):.4f} ms")
print(f"  1023:      {event(lambda: a2 @ b2):.4f} ms")
print(f"  1024 hot:  {event(lambda: a @ b):.4f} ms")
print(f"  1024 hot:  {event(lambda: a @ b):.4f} ms")
print(f"  1024 hot:  {event(lambda: a @ b):.4f} ms")

print("\n=== Test D: Cold 1024 fp32, then 512 fp32, then 1024 fp32 again ===")
a = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
b = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
print(f"  1024 cold: {event(lambda: a @ b):.4f} ms")
print(f"  1024 cold: {event(lambda: a @ b):.4f} ms")
a2 = torch.randn(512, 512, device='mps', dtype=torch.float32)
b2 = torch.randn(512, 512, device='mps', dtype=torch.float32)
print(f"   512:      {event(lambda: a2 @ b2):.4f} ms")
print(f"  1024 ?:    {event(lambda: a @ b):.4f} ms")
print(f"  1024 ?:    {event(lambda: a @ b):.4f} ms")
print(f"  1024 ?:    {event(lambda: a @ b):.4f} ms")

print("\n=== Test E: Cold 1024 fp32, then 1025 fp32 (off-by-one), then 1024 fp32 ===")
a = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
b = torch.randn(1024, 1024, device='mps', dtype=torch.float32)
print(f"  1024 cold: {event(lambda: a @ b):.4f} ms")
print(f"  1024 cold: {event(lambda: a @ b):.4f} ms")
a2 = torch.randn(1025, 1025, device='mps', dtype=torch.float32)
b2 = torch.randn(1025, 1025, device='mps', dtype=torch.float32)
print(f"  1025:      {event(lambda: a2 @ b2):.4f} ms")
print(f"  1024 ?:    {event(lambda: a @ b):.4f} ms")
print(f"  1024 ?:    {event(lambda: a @ b):.4f} ms")
print(f"  1024 ?:    {event(lambda: a @ b):.4f} ms")
