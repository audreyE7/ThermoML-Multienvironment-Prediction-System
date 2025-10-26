import numpy as np
import pandas as pd

ENV_ENUM = {0: "desert", 1: "ocean", 2: "vacuum", 3: "space"}

def compute_alpha(k, rho, cp):
    return np.asarray(k) / (np.asarray(rho) * np.asarray(cp))

def add_dimensionless(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Bi, Fo and simple placeholders for Nu via environment cues.
    Assumes columns: k, rho, cp, h, Lc, t, alpha(optional)
    """
    out = df.copy()
    if "alpha" not in out.columns:
        out["alpha"] = compute_alpha(out["k"], out["rho"], out["cp"])

    # Characteristic length default if missing
    out["Lc"] = out.get("Lc", pd.Series(np.full(len(out), 0.01), index=out.index))

    # Biot and Fourier
    out["Bi"] = (out["h"] * out["Lc"]) / out["k"]
    out["Fo"] = (out["alpha"] * out["t"]) / (out["Lc"] ** 2)

    # Very simple Nu proxy by environment (you’ll replace with your correlations)
    env = out.get("env_code", pd.Series(np.zeros(len(out)), index=out.index))
    # baseline Nusselt guess (convective relevance): vacuum/space very low, ocean higher
    base_nu = np.select(
        [env == 1, env == 0, env == 2, env == 3],
        [20.0, 5.0, 1.0, 1.0],
        default=5.0,
    )
    out["Nu"] = base_nu * (1 + 0.1 * out["Bi"].clip(upper=1.0))  # light coupling
    return out

FEATURE_COLUMNS = [
    "k","rho","cp","epsilon","h","q_in","T_env","Lc","t","alpha","Bi","Fo","Nu","env_code"
]
TARGET_COLUMN = "T_max"
