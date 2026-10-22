// Neutral queries do not commit a measurement, elapsed time or season crossing.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
extern "C" void pregame_neutral(long n, long np, const int64_t* id,
 const double* day, const double* season, const double* exposure,
 const double* obs, const double* init, const double* shift,
 const double* transient, const double* pars, double mean,
 double* pred, double* predvar, double* post, double* postvar,
 double* prior_exposure, double* last_observed) {
 std::vector<double> mu(np,0),d(np,0),pmm(np,0),pmd(np,0),pdd(np,0),
  last(np,0),lastseason(np,0),mass(np,0);
 std::vector<bool> seen(np,false);
 for(long i=0;i<n;){
  const long p=id[i];long end=i+1;
  while(end<n && id[end]==p && day[end]==day[i])++end;
  double pm=init[i],pd=0,amm=pars[1],amd=0,add=pars[3];
  if(seen[p]){
   const double gap=std::max(1.0,day[i]-last[p]);
   const double sc=season[i]>lastseason[p],a=std::exp(-gap/pars[0]),b=std::exp(-gap/pars[2]);
   const double anchor=mean+shift[i]*sc;
   pm=anchor+a*(mu[p]-anchor);pd=b*d[p];
   amm=a*a*pmm[p]+pars[1]*(1-a*a)+pars[4]*sc;
   amd=a*b*pmd[p];add=b*b*pdd[p]+pars[3]*(1-b*b);
  }
  const double level=pm+pd,hph=amm+2*amd+add;
  double w=0,weighted=0;
  for(long j=i;j<end;++j){
   pred[j]=level;predvar[j]=hph;prior_exposure[j]=mass[p];last_observed[j]=seen[p]?last[p]:-1;
   if(exposure[j]>0 && std::isfinite(obs[j])){w+=exposure[j];weighted+=exposure[j]*(obs[j]-transient[j]);}
  }
  double after=level,aftervar=hph;
  if(w>0){
   const double s=hph+pars[5]/w,c0=amm+amd,c1=amd+add,km=c0/s,kd=c1/s;
   const double innovation=weighted/w-level;
   mu[p]=pm+km*innovation;d[p]=pd+kd*innovation;
   pmm[p]=amm-km*c0;pmd[p]=amd-km*c1;pdd[p]=add-kd*c1;
   last[p]=day[i];lastseason[p]=season[i];seen[p]=true;mass[p]+=w;
   after=mu[p]+d[p];aftervar=pmm[p]+2*pmd[p]+pdd[p];
  }
  for(long j=i;j<end;++j){post[j]=after;postvar[j]=aftervar;}
  i=end;
 }
}
