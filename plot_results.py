import numpy as np
import matplotlib.pyplot as plt

# Measured values from the three runs (N = 4096 transpose)
runs        = ["Run 1", "Run 2", "Run 3"]
uncoalesced = [160.4, 167.1, 167.1]   # GB/s
coalesced   = [324.1, 323.6, 323.0]   # GB/s

x = np.arange(len(runs))
w = 0.36

fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w/2, uncoalesced, w, label="Uncoalesced", color="#e4572e")
b2 = ax.bar(x + w/2, coalesced,   w, label="Coalesced (shared-memory tile)", color="#2e86ab")

# value labels on bars
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 4,
                f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)

# average lines
ax.axhline(np.mean(uncoalesced), color="#e4572e", ls="--", lw=1, alpha=0.7)
ax.axhline(np.mean(coalesced),   color="#2e86ab", ls="--", lw=1, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.set_ylabel("Effective bandwidth (GB/s)")
ax.set_title("Matrix Transpose (N = 4096): Uncoalesced vs Coalesced\nNVIDIA GeForce RTX 5060 Ti")
ax.set_ylim(0, max(coalesced) * 1.28)
ax.legend(loc="upper center", ncol=2)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("screenshots/transpose_bandwidth.png", dpi=200)
plt.show()