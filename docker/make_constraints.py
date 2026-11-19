"""Write docker/constraints.txt from the versions recorded in a trained model bundle.

    python docker/make_constraints.py                          # uses models/embassy_latest
    python docker/make_constraints.py models/embassy_latest
Run it again after every retraining, then `docker compose build`.
"""
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[1] if len(sys.argv) > 1 else "models/embassy_latest")
v = json.loads((bundle / "metadata.json").read_text())["versions"]
lines = [f"lightgbm=={v['lightgbm']}", f"scikit-learn=={v['scikit_learn']}", f"pandas=={v['pandas']}"]
out = Path(__file__).with_name("constraints.txt")
out.write_text("# made by docker/make_constraints.py from " + str(bundle) + "\n" + "\n".join(lines) + "\n")
print(f"wrote {out}:\n" + "\n".join(lines) + f"\n(model was trained with Python {v['python']}; the image uses 3.11)")
