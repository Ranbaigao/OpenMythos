import time
it = iter(loader)   # 用你脚本里构造好的 loader
t0 = time.perf_counter()
for _ in range(256):
    next(it)
print(f"{256*1024 / (time.perf_counter()-t0):.0f} tok/s (dataloader only)")