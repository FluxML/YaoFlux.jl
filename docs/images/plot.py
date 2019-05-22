#!/usr/bin/env python
import fire
from plotlib import *

class PLT(object):
    def fig1(self, tp='pdf'):
        nsite = 4
        data = np.loadtxt("../data/loss_history_%d.dat"%nsite)
        EG = -8
        with DataPlt(figsize=(5,4), filename="fig1.%s"%tp) as dp:
            plt.plot(np.arange(len(data)), data/nsite, lw=2)
            plt.xlabel("Training Step")
            plt.ylabel("Energy/Site")
            plt.axhline(EG/nsite, ls='--', lw=2)
            plt.tight_layout()
            plt.xlim(0,200)

fire.Fire(PLT())
