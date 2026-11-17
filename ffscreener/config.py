import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(os.environ.get("FFSCREENER_CACHE", ROOT / "data_cache"))
CACHE.mkdir(parents=True, exist_ok=True)
(CACHE / "sec").mkdir(exist_ok=True)

# SEC exige un User-Agent identificable. Definí FFSCREENER_CONTACT="Tu Nombre tu@mail".
CONTACT = os.environ.get("FFSCREENER_CONTACT", "ffscreener-research research@example.org")
SEC_HEADERS = {"User-Agent": CONTACT}

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
PORTFOLIOS_YAML = ROOT / "ffscreener" / "portfolios.yaml"
PCTS = list(range(5, 101, 5))  # French publica cada 5to percentil NYSE
