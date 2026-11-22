"""Render final-paper figures from freshly computed local outputs."""
from pathlib import Path
import subprocess
import sys

def main():
    folder=Path(__file__).resolve().parent
    for script in ['export_figure_data.py','plot_encoding.py','plot_text_glm.py','plot_cohort.py','build_tables.py']:
        subprocess.run([sys.executable,str(folder/script)],check=True)

if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
