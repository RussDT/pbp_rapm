// Independent two-state OU replay with query-only missing/zero-exposure games.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
extern "C" void skill_history(long n, long np, const int64_t* player,
 const double* day, const double* season, const double* exposure, const double* y,
 const double* pars, double mean, int pct,
 double* prediction, double* post, double* variance, double* prior_exposure,
 double* prior_count, double* last_observed_day) {
 std::vector<double> mu(np,0), dev(np,0), pmm(np,0), pmd(np,0), pdd(np,0),
  last(np,0), prevseason(np,0), seen_exposure(np,0), count(np,0), observed(np,-1);
 std::vector<bool> seen(np,false);
 auto bound=[pct](double x){return pct ? std::min(1.0,std::max(0.0,x)) : std::max(0.0,x);};
 for(long i=0;i<n;){
  long p=player[i];double pm,pd,amm,amd,add;
  long end=i+1;
  while(end<n && player[end]==p && day[end]==day[i]) ++end;
  if(!seen[p]){pm=mean;pd=0;amm=pars[1];amd=0;add=pars[3];seen[p]=true;}
  else {
   double dt=std::max(0.0,day[i]-last[p]);
   double a=std::exp(-dt/pars[0]),b=std::exp(-dt/pars[2]);
   pm=mean+a*(mu[p]-mean);pd=b*dev[p];
   amm=a*a*pmm[p]+pars[1]*(1-a*a)+pars[4]*(season[i]>prevseason[p]);
   amd=a*b*pmd[p];add=b*b*pdd[p]+pars[3]*(1-b*b);
  }
  double latent=pm+pd,hph=std::max(0.0,amm+2*amd+add);
  double weight=0,weighted_y=0,observations=0;
  for(long j=i;j<end;++j){
   prediction[j]=bound(latent);variance[j]=hph;
   prior_exposure[j]=seen_exposure[p];prior_count[j]=count[p];last_observed_day[j]=observed[p];
   if(exposure[j]>0 && std::isfinite(y[j])){weight+=exposure[j];weighted_y+=exposure[j]*y[j];observations+=1;}
  }
  if(weight>0){
   double s=hph+pars[5]/weight,c0=amm+amd,c1=amd+add;
   double km=c0/s,kd=c1/s,innovation=weighted_y/weight-latent;
   mu[p]=pm+km*innovation;dev[p]=pd+kd*innovation;
   pmm[p]=amm-km*c0;pmd[p]=amd-km*c1;pdd[p]=add-kd*c1;
   seen_exposure[p]+=weight;count[p]+=observations;observed[p]=day[i];
  } else {mu[p]=pm;dev[p]=pd;pmm[p]=amm;pmd[p]=amd;pdd[p]=add;}
  for(long j=i;j<end;++j) post[j]=bound(mu[p]+dev[p]);
  last[p]=day[i];prevseason[p]=season[i];i=end;
 }
}
