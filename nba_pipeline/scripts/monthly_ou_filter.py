"""Checked compiled implementation of the existing two-state OU filter."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np

SOURCE=Path(__file__).with_suffix('.cpp').with_name('monthly_ou_kernel.cpp')


def load_kernel(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    output=directory/f'monthly_ou_{digest}.so'
    if not output.exists():
        subprocess.run(['c++','-O3','-std=c++11','-shared','-fPIC',str(SOURCE),'-o',str(output)],check=True)
    lib=ctypes.CDLL(str(output.resolve()))
    fn=lib.monthly_ou
    d=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags='C_CONTIGUOUS')
    ids=np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_long,ctypes.c_long,ids,d,d,d,d,d,ctypes.c_double,
                 d,d,d,d,ctypes.c_double,d,d,d,d]
    fn.restype=None
    return fn


class FastOU:
    def __init__(self,sub,cache):
        if not sub.groupby('player_idx',sort=False).game_date.apply(lambda d:d.is_monotonic_increasing).all():
            raise ValueError('OU rows must be chronological within player')
        self.kernel=load_kernel(cache);self.n=len(sub)
        self.ids=np.ascontiguousarray(sub.player_idx,dtype=np.int64)
        self.nplayers=int(self.ids.max())+1
        self.dt=np.ascontiguousarray(np.maximum(sub.days_gap,1),dtype=float)
        self.sc=np.ascontiguousarray(sub.season_cross,dtype=float)
        self.expo=np.ascontiguousarray(sub.exposure,dtype=float)
        self.zero=np.zeros(self.n)

    def run(self,obs,pars,mean,init=None,shift=None,trans=None,context=None,lam=0):
        output=[np.empty(self.n) for _ in range(4)]
        arrays=[np.ascontiguousarray(x,dtype=float) for x in [obs,pars,
            np.full(self.n,mean) if init is None else init,
            self.zero if shift is None else shift,self.zero if trans is None else trans,
            self.zero if context is None else context]]
        self.kernel(self.n,self.nplayers,self.ids,self.dt,self.sc,self.expo,
                    arrays[0],arrays[1],float(mean),*arrays[2:],float(lam),*output)
        return output
