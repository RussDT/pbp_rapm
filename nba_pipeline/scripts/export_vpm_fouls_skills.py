"""Append measured foul-drawing states without changing existing tracking inputs."""
from export_vpm_tracking_skills import ROOT,build
LANES={s:('fouls_drawn_next_game/'+s,'own_'+s+'100',s+'_valid',s+'_reason','offensive' if s=='pfd' else 'defensive') for s in ['off_fouls_drawn','charges_drawn','pfd']}
if __name__=='__main__':build(ROOT/'results/estimated_skills/vpm_tracking_v1',ROOT/'results/estimated_skills/vpm_fouls_v1',LANES)
