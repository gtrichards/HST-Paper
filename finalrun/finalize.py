"""
Rebuild everything in pipeline_output/final/ after a batch, in the right order:

    check_redshift.py     NED/SIMBAD check for every worked object (cached)
    pass_decisions.py     the working ledger and one final figure per object
    build_paper_table.py  the paper table in the xlsx layout, all 474 rows

One command so the three never drift out of step.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for script in ("check_redshift.py", "pass_decisions.py", "build_paper_table.py"):
    print("\n=== %s ===" % script, flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, script)])
    if r.returncode:
        sys.exit("%s failed" % script)
