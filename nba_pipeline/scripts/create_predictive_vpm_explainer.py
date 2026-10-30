"""Three-page visual VPM methods note. Examples schematic; results source-backed.

Run with bundled Python and python-docx/matplotlib/numpy. Desktop staging:
PYTHONPATH=tmp/vpm-figure-deps <bundled-python> <this-file>
Render using the bundled document renderer. Source receipts are in ../vpm.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Arc
from matplotlib import font_manager
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/documents'
FIG = ROOT/'tmp/vpm-explainer-figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
font_path = Path('/System/Library/Fonts/Avenir Next.ttc')
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
    FAMILY = font_manager.FontProperties(fname=str(font_path)).get_name()
else:
    FAMILY = 'DejaVu Sans'
INK, GREY, LINE = '#15252E', '#53646D', '#CED6D9'
BLUE, ORANGE, PALE = '#215C78', '#C64B2B', '#EDF2F3'
plt.rcParams.update({'font.family': FAMILY, 'font.size': 11,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})


def base(h):
    fig, ax = plt.subplots(figsize=(7,h))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,7),ylim=(0,h))
    ax.axis('off')
    return fig,ax


def t(ax,x,y,s,size=11,c=INK,bold=False,ha='left',**kw):
    return ax.text(x,y,s,fontsize=size,color=c,weight='bold' if bold else 'normal',
                   ha=ha,va='center',zorder=5,**kw)


def arrow(ax,a,b,c=LINE,lw=1.2,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,
        lw=lw,color=c,connectionstyle=f'arc3,rad={rad}'))


def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)


# Court and scoreboard: one team outcome is shared by ten players.
fig,ax=base(2.30)
x0,y0,w,h=.08,.40,4.02,1.80
ax.add_patch(Rectangle((x0,y0),w,h,fill=False,ec=LINE,lw=1.1))
ax.plot([x0+w/2]*2,[y0,y0+h],color=LINE,lw=1)
ax.add_patch(Circle((x0+w/2,y0+h/2),.29,fill=False,ec=LINE,lw=1))
for left in [True,False]:
    bx=x0 if left else x0+w-.72
    ax.add_patch(Rectangle((bx,y0+.46),.72,.88,fill=False,ec=LINE,lw=1))
    cx=bx+.72 if left else bx
    ax.add_patch(Arc((cx,y0+h/2),.88,.88,theta1=-90 if left else 90,
                     theta2=90 if left else 270,ec=LINE,lw=1))
    rimx=x0+.17 if left else x0+w-.17
    ax.add_patch(Circle((rimx,y0+h/2),.055,fill=False,ec=LINE,lw=1))
positions=[(2.10,1.65),(2.75,1.95),(3.65,1.83),(2.72,.70),(3.65,.78)]
for i,(x,y) in enumerate(positions):
    ax.add_patch(Circle((x,y),.102,color=ORANGE,zorder=4))
    t(ax,x,y,str(i+1),7,'white',True,'center')
for i,(x,y) in enumerate([(2.35,1.48),(2.95,1.72),(3.42,1.57),(2.98,.93),(3.43,1.06)]):
    ax.add_patch(Circle((x,y),.102,fc='white',ec=BLUE,lw=1.5,zorder=4))
    t(ax,x,y,str(i+1),7,BLUE,True,'center')
t(ax,.10,.16,'10 players on court',10,GREY)
arrow(ax,(4.20,1.30),(4.64,1.30),INK)
t(ax,5.80,2.09,'ONE SHARED SCOREBOARD',8.5,GREY,True,'center')
t(ax,5.14,1.65,'112',31,INK,True,'center')
t(ax,6.43,1.65,'106',31,INK,True,'center')
t(ax,5.80,1.65,'-',19,GREY,ha='center')
t(ax,5.14,1.23,'TEAM A',8,GREY,ha='center')
t(ax,6.43,1.23,'TEAM B',8,GREY,ha='center')
t(ax,5.80,.71,'+6',35,ORANGE,True,'center')
t(ax,5.80,.24,'team scoring margin',10,GREY,ha='center')
save(fig,'scoreboard')

# Same units as the scoreboard, now an illustrative average-player comparison.
fig,ax=base(1.45)
t(ax,.05,1.30,'AVERAGE PLAYER',8.5,GREY,True)
t(ax,3.42,1.30,'ILLUSTRATIVE +4 PLAYER',8.5,GREY,True)
ax.plot([1.08,3.75],[.74,1.10],color=ORANGE,lw=2.2)
ax.plot([1.08,3.75],[.74,.62],color=BLUE,lw=2.2)
ax.scatter([1.08],[.74],s=48,color=INK,zorder=4)
ax.scatter([3.75,3.75],[1.10,.62],s=42,color=[ORANGE,BLUE],zorder=4)
t(ax,.06,.73,'100\n100',14,INK,True,linespacing=.95)
t(ax,3.93,1.10,'103 scored',12,ORANGE,True)
t(ax,3.93,.62,'99 allowed',12,BLUE,True)
ax.plot([5.45,5.56,5.56,5.45],[1.10,1.10,.62,.62],color=INK,lw=1.2)
t(ax,6.24,.80,'+4',30,INK,True,'center')
t(ax,6.24,.25,'point margin',9,GREY,ha='center')
t(ax,1.08,.18,'Even scoreboard',9,GREY,ha='center')
save(fig,'addition')

# Explicit published training windows and the actual future-possession target.
fig,ax=base(3.23)
t(ax,.02,3.02,'EPM',18,INK,True)
t(ax,1.49,3.07,'20-season box-score SPM',11,INK,True)
t(ax,4.48,3.07,'6-season tracking SPM',10.5,INK,True)
t(ax,1.49,2.77,'Statistics paired with RAPM from the same historical seasons',9,GREY)
t(ax,.02,2.29,'VPM',18,ORANGE,True)
t(ax,1.49,2.38,'Predictive box-score / advanced stats',10,ORANGE,True)
ax.add_patch(Circle((1.61,1.99),.075,color=ORANGE))
ax.plot([1.61]*2,[1.80,2.22],color=ORANGE,lw=1)
arrow(ax,(1.80,1.99),(6.50,1.99),BLUE,1.8)
t(ax,4.20,2.22,'FOLLOWING 365 DAYS',9,BLUE,True,'center')
t(ax,1.49,1.62,'EACH TRAINING ROW',8.5,GREY,True)
t(ax,3.00,1.30,'5 on offense',9,ORANGE,True,'center')
t(ax,4.73,1.30,'5 on defense',9,BLUE,True,'center')
t(ax,6.30,1.30,'Actual points',9,INK,True,'center')
for y,label,points in [(1.02,'Possession A','0'),(.68,'Possession B','2'),(.34,'Possession C','3')]:
    t(ax,1.49,y,label,8.5,GREY)
    for j in range(5):
        ax.add_patch(Circle((2.62+j*.19,y),.043,color=ORANGE))
        ax.add_patch(Circle((4.35+j*.19,y),.043,fc='white',ec=BLUE,lw=1))
    t(ax,6.30,y,points,15,INK,True,'center')
t(ax,4.25,.08,'Player identities + observed possession outcome',9,GREY,ha='center')
fig.set_size_inches(7,2.64)
save(fig,'training')

# Recency is a different clock from the one-year forward training window.
fig,ax=base(.95)
x=np.linspace(0,4,30)
y=2**((x-4)*365.25/550)
for xx,yy in zip(.18+x*1.53,y):
    ax.plot([xx]*2,[.22,.22+.55*yy],color=BLUE,lw=3,solid_capstyle='butt')
ax.plot([.16,6.32],[.21,.21],color=LINE,lw=.8)
t(ax,.16,.07,'Older lineup possessions',9,GREY)
t(ax,6.34,.07,'Most recent',9,BLUE,True,'right')
t(ax,.16,.79,'LESS WEIGHT',8,GREY,True)
t(ax,6.34,.87,'MORE WEIGHT',8,BLUE,True,'right')
save(fig,'recency')

# Hierarchical decomposition, without implying widths are quantitative shares.
fig,ax=base(1.75)
t(ax,.08,1.57,'ON EACH SIDE OF THE BALL',8.5,GREY,True)
t(ax,2.12,1.18,'Shot value',12,INK,True,'center')
t(ax,4.90,1.18,'Turnovers',12,INK,True,'center')
t(ax,6.26,1.18,'Second chances',11,INK,True,'center')
t(ax,4.18,1.18,'+',14,GREY,ha='center')
t(ax,5.56,1.18,'+',14,GREY,ha='center')
ax.plot([2.12,2.12],[1.00,.82],color=LINE,lw=1)
ax.plot([.46,3.77],[.82,.82],color=LINE,lw=1)
for xx,label,sub in [(.46,'Rim','Mix +\nconversion'),(1.57,'Mid / ref','SQ +\nresidual'),
                      (2.67,'Threes','Mix + SQ\n+ residual'),(3.77,'Free throws','Actual\nvalue')]:
    ax.plot([xx]*2,[.82,.67],color=LINE,lw=1)
    t(ax,xx,.50,label,9.5,BLUE,True,'center')
    t(ax,xx,.18,sub,8,GREY,ha='center',linespacing=1.1)
t(ax,4.90,.58,'Keep or delete\npossessions',9,GREY,ha='center',linespacing=1.2)
t(ax,6.26,.58,'Value after\nthe first chance',9,GREY,ha='center',linespacing=1.2)
save(fig,'lanes')

# One VPM model and one benchmark, on the same historical observations.
receipt=json.loads((ROOT.parent/'vpm/vpm/evidence/net_lane_clean_summary.json').read_text())
score={(r['system'],r['test']):r for r in receipt['scorecard']
       if r['period']=='stress_2023_2026' and r['system'] in ('dmap_a100','EPM')}
metrics=[
    ('game_margin','Game margin','Points',1,13.85,14.00,[13.85,13.90,13.95,14.00]),
    ('team_net_rating','Team net rating','Points per 100 possessions',1,3.0,3.2,[3.0,3.1,3.2]),
    ('team_win_pct','Team win percentage','Percentage points',100,8.3,8.7,[8.3,8.5,8.7]),
    ('team_win_pct_low_continuity','Roster-change team wins','Percentage points; 13 team-seasons',100,10.8,11.8,[10.8,11.3,11.8]),
]
fig,ax=base(3.62)
t(ax,5.95,3.42,'VPM',9,ORANGE,True,'right')
t(ax,6.94,3.42,'EPM',9,BLUE,True,'right')
for i,(key,label,unit,scale,lo,hi,ticks) in enumerate(metrics):
    y=3.05-i*.82
    def px(v):return 2.78+(v-lo)/(hi-lo)*2.20
    v=score[('dmap_a100',key)]['rmse']*scale
    e=score[('EPM',key)]['rmse']*scale
    t(ax,.04,y+.04,label,10.5,INK,True)
    t(ax,.04,y-.19,unit,8.1,GREY)
    ax.plot([2.78,4.98],[y,y],color=LINE,lw=.7)
    ax.plot([px(v),px(e)],[y,y],color=GREY,lw=1.2)
    ax.scatter([px(v)],[y],s=48,color=ORANGE,zorder=4)
    ax.scatter([px(e)],[y],s=48,fc='white',ec=BLUE,lw=1.6,zorder=4)
    t(ax,5.95,y,f'{v:.3f}',11,ORANGE,True,'right')
    t(ax,6.94,y,f'{e:.3f}',11,BLUE,False,'right')
    for tick in ticks:
        t(ax,px(tick),y-.25,f'{tick:.2f}' if key=='game_margin' else f'{tick:.1f}',7.8,GREY,ha='center')
save(fig,'results')


doc=Document();sec=doc.sections[0]
sec.page_width,sec.page_height=Inches(8.5),Inches(11)
sec.top_margin,sec.bottom_margin=Inches(.52),Inches(.50)
sec.left_margin=sec.right_margin=Inches(.75)
sec.footer_distance=Inches(.22)
for name in ['Normal','Title','Subtitle','Heading 1','Caption']:
    s=doc.styles[name];s.font.name='Carlito';s.font.color.rgb=RGBColor(0,0,0)
    for fonts in s.element.xpath('.//w:rFonts'):
        for attr in list(fonts.attrib):
            if 'theme' in attr.lower():del fonts.attrib[attr]
    for border in s.element.xpath('.//w:pBdr'):border.getparent().remove(border)
s=doc.styles['Normal'];s.font.size=Pt(11)
s.paragraph_format.line_spacing=1.08;s.paragraph_format.space_after=Pt(7)
s=doc.styles['Title'];s.font.name='Noto Serif';s.font.size=Pt(27);s.font.bold=False
s.paragraph_format.space_before=Pt(0);s.paragraph_format.space_after=Pt(8)
s.paragraph_format.keep_with_next=True
s=doc.styles['Heading 1'];s.font.size=Pt(13);s.font.bold=True
s.paragraph_format.space_before=Pt(9);s.paragraph_format.space_after=Pt(4)
s.paragraph_format.keep_with_next=True
s=doc.styles['Caption'];s.font.size=Pt(9);s.font.bold=False;s.font.italic=False
s.font.color.rgb=RGBColor.from_string('53646D')
s.paragraph_format.line_spacing=1.0;s.paragraph_format.space_after=Pt(7)


def p(text,style='Normal'):
    q=doc.add_paragraph(text,style);q.paragraph_format.widow_control=True;return q


def kicker(text):
    q=p(text,'Caption');q.paragraph_format.space_after=Pt(4)
    q.runs[0].font.size=Pt(8);q.runs[0].font.bold=True


def figure(name,alt,caption=None):
    q=doc.add_paragraph();q.paragraph_format.space_after=Pt(2)
    q.paragraph_format.keep_with_next=bool(caption)
    pic=q.add_run().add_picture(str(FIG/f'{name}.png'),width=Inches(7))
    pic._inline.docPr.set('descr',alt)
    if caption:p(caption,'Caption')


kicker('01   /   THE QUESTION')
p('From the scoreboard to player impact','Title')
p('Basketball begins with a simple outcome: score more points than the opponent. A player helps by creating points for their team or preventing points at the other end. An impact metric estimates how much that player changes the margin.')
figure('scoreboard','Schematic court with five offensive and five defensive players. An illustrative final score of 112 to 106 produces a shared team margin of plus six points.',
       'Illustrative game and player positions. Ten players share the outcome; the final score does not tell us each player’s contribution.')
p('Put different contributions on the same scale','Heading 1')
p('VPM expresses impact in points per 100 possessions, relative to an average player. A +3 offensive player with +1 defensive impact improves the expected margin by four points. Positive defense means fewer points allowed.')
figure('addition','Illustrative equal teams score 100 and allow 100 per 100 possessions. Replacing an average player with a plus three offensive, plus one defensive player changes the example to 103 scored and 99 allowed, a plus four margin.',
       'Illustrative comparison holding other conditions fixed. VPM is a rate; playing time determines how much of it reaches a team forecast.')
p('Separate the player from the situation','Heading 1')
p('A good team’s margin cannot simply be credited equally to everyone. Players share teammates, face different opponents and play different roles. We combine predictive box-score and advanced stats with lineup evidence to estimate the contribution that is likely to carry forward.')
p('We build on DARKO’s ideas for forecasting player statistics and EPM’s combination of a statistical expectation with adjusted plus-minus. Our central training difference is the time relationship: we learn how information available at a date predicts actual lineup scoring in the following year. [1–3]')

doc.add_page_break()
kicker('02   /   THE METHOD')
p('Train on what happens next','Title')
p('EPM’s published SPM models learn from multi-season RAPM. Our training target is the actual number of points scored on each possession in the following year, using all ten player identities to separate offensive and defensive impact. [2, 3]')
figure('training','EPM publishes a 20-season box-score SPM and six-season tracking SPM, pairing statistics with RAPM from the same years. VPM trains against actual scoring on possessions in the following 365 days. Each row identifies five offensive players, five defensive players and actual points scored. Three illustrative outcomes are zero, two and three points.',
       'Illustrative possession rows. EPM windows: 2003–04 to 2022–23 for box-score SPM; 2017–18 to 2022–23 for tracking SPM. Both systems produce predictive ratings.')
p('We fit a regularized lineup model directly to those future possession outcomes. It learns how the earlier predictive stats relate to impact, allowing player-specific deviations. We pool many completed forward years; a cohort enters training only after its full year has finished.')
p('The final rating keeps a memory of past lineups','Heading 1')
figure('recency','The selected model weights older lineup possessions less and recent possessions more, with a smooth exponential weighting curve shown as vertical bars.',
       'This recency weighting is separate from the one-year forward training horizon. Past evidence still contributes; recent evidence has more influence.')
p('The same rating has an exact decomposition','Heading 1')
figure('lanes','On offense and defense, shot value plus turnover value plus second-chance value equals that side’s impact. Shot value separates into rim mix and conversion; midrange and reference with ShotQuality and residual; threes with mix, ShotQuality and residual; and actual free-throw value.')
p('ShotQuality separates expected shooting value from performance above or below expectation; we shrink the noisier residual more strongly. Midrange also carries the reference used to value shot mix. All lanes add back to VPM, and describe lineup impact rather than unique credit for a player’s own events. [4, 5]','Caption')

doc.add_page_break()
kicker('03   /   THE EVIDENCE')
p('Measured against future games','Title')
p('Our historical 2023–26 tests cover 4,920 games and 120 team-seasons. Seven checks cover game margins, team results, roster changes and veterans. Root mean squared error (RMSE) measures game-margin error in scoreboard points; lower is better. [3, 4]')
figure('results','Selected VPM versus EPM historical RMSE: game margin 13.888 versus 13.942 points; team net rating 3.096 versus 3.120 points per 100 possessions; team win percentage 8.412 versus 8.514 percentage points; roster-change team win percentage 11.030 versus 11.508 percentage points on 13 team-seasons. Lower is better. Filled orange points indicate VPM, open blue points indicate EPM. Each row has its own focused scale.',
       'One VPM model versus EPM. Each row has its own units and focused scale. Points are error estimates, not uncertainty intervals; lower is better.')
p('What these numbers establish','Heading 1')
p('VPM has lower error point estimates in these comparisons. The game-margin difference is about 0.053 scoreboard points of RMSE, and the team net-rating difference is about 0.024 points per 100 possessions. These are small gaps; they do not establish predictive superiority over EPM.')
p('What remains to be tested','Heading 1')
p('The games use common actual future minutes and pace. That tests player quality given participation, rather than a complete pregame availability forecast. Roster-change team results are a particularly small sample: 13 team-seasons.')
p('Historical method searches and source-availability limitations remain. These seasons are already inspected research evidence. The next untouched evaluation is 2026–27. Our forward training target is a methodological distinction, not proof that the method is better.')
p('Sources: [1] darko.app/about. [2] dunksandthrees.com/about/epm. VPM notes: [3] Expanded forward priors. [4] Exact Net lanes and Defensive Parent calibration, clean-source scorecard. [5] Current Parent ShotQuality ablation. Model snapshot September 2026; updated October 21, 2026. Automated live refresh remains unfinished.','Caption')

for lang in doc.styles.element.xpath('.//w:lang'):lang.set(qn('w:val'),'en-US')
doc.core_properties.title='How we built predictive VPM'
doc.core_properties.author='Russell Thomas'
doc.core_properties.subject='Scoreboard, forward training, exact decomposition and historical evidence'
q=sec.footer.paragraphs[0];q.alignment=2
q.add_run('PREDICTIVE VPM     /     ')
f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');q._p.append(f)
for r in q.runs:r.font.name='Carlito';r.font.size=Pt(8);r.font.color.rgb=RGBColor.from_string('53646D')
path=OUT/'predictive_vpm_three_pager.docx';doc.save(path)
print(path)
print('Prose words',sum(len(q.text.split()) for q in doc.paragraphs))
