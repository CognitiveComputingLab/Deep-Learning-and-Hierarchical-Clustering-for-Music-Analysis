import pitchscapes.reader as rd
import pitchscapes.plotting as pt
import matplotlib.pyplot as plt

scape = rd.get_pitch_scape(r"n11op95_01.mid")
fig, ax = plt.subplots(figsize=(14, 7))
pt.key_scape_plot(scape=scape, n_samples=100, ax=ax)
print("xlim:", ax.get_xlim())
print("ylim:", ax.get_ylim())