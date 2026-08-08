#!/usr/bin/env python
'''Generate the tonic-relative weighted exact-DP tree Pitch Scape.'''

from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from dp_pitchscape import visualization_cli


if __name__=='__main__':
    visualization_cli(
        'tonic_weighted','dptree_tonic_weighted.png')
