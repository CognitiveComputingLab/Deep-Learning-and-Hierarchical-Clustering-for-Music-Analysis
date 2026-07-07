import sys
sys.path.insert(0, 'src')

import pitchscapes.reader as rd
import pitchscapes.plotting as pt
import matplotlib.pyplot as plt
from fugue_loader import load_fugue_dez

midi_path = r"external\pitchscapes-repo\doc\Prelude_No_1_BWV_846_in_C_Major.mid"
dez_path = r"external\algomus-data\fugues\bach-wtc-i\01-bwv846-ref.dez"

# load pitch scape
scape = rd.get_pitch_scape(midi_path)
fig, ax = plt.subplots(figsize=(10, 6))
pt.key_scape_plot(scape=scape, n_samples=100, ax=ax)

# load Algomus annotation
tree = load_fugue_dez(dez_path)
print("Fugue tree:")
print(tree)  

plt.savefig("bwv846_scape.png", dpi=120, bbox_inches='tight')
plt.show()