// Pure terminal queries of frozen full_skill_history_kernel.cpp source states.
// Synthetic queries never commit seen/last/season/covariance; real missing or
// zero-opportunity source rows retain the original update/time semantics.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
extern "C" void daily_skill_queries(long n,long np,const int64_t* player,
 const double* day,const double* season,const double* exposure,const double* y,
 const int64_t* query,const double* pars,double mean,int pct,
 double* prediction,double* prior_exposure,double* last_observed_day) {
 std::vector<double> mu(np,0),dev(np,0),pmm(np,0),pmd(np,0),pdd(np,0),last(np,0),
  prevseason(np,0),seen_exposure(np,0),observed(np,-1);
 std::vector<bool> seen(np,false);
 auto bound=[pct](double x){return pct?std::min(1.0,std::max(0.0,x)):std::max(0.0,x);};
 for(long i=0;i<n;){
  long p=player[i],end=i+1;while(end<n&&player[end]==p&&day[end]==day[i])++end;
  // A date can contain a source batch and multiple pure queries. Source
  // propagation uses source season, exactly as the frozen source-only kernel.
  long source=-1;for(long j=i;j<end;++j)if(query[j]<0){source=j;break;}
  auto propagate=[&](double s,double& pm,double& pd,double& amm,double& amd,double& add){
   if(!seen[p]){pm=mean;pd=0;amm=pars[1];amd=0;add=pars[3];}
   else{double dt=std::max(0.0,day[i]-last[p]);double a=std::exp(-dt/pars[0]),b=std::exp(-dt/pars[2]);
    pm=mean+a*(mu[p]-mean);pd=b*dev[p];amm=a*a*pmm[p]+pars[1]*(1-a*a)+pars[4]*(s>prevseason[p]);
    amd=a*b*pmd[p];add=b*b*pdd[p]+pars[3]*(1-b*b);}
  };
  for(long j=i;j<end;++j)if(query[j]>=0){
   double pm,pd,amm,amd,add;propagate(season[j],pm,pd,amm,amd,add);long q=query[j];
   prediction[q]=bound(pm+pd);prior_exposure[q]=seen_exposure[p];last_observed_day[q]=observed[p];
  }
  if(source>=0){
   double pm,pd,amm,amd,add;propagate(season[source],pm,pd,amm,amd,add);
   double latent=pm+pd,hph=std::max(0.0,amm+2*amd+add),weight=0,weighted_y=0;
   for(long j=i;j<end;++j)if(query[j]<0&&exposure[j]>0&&std::isfinite(y[j])){
    weight+=exposure[j];weighted_y+=exposure[j]*y[j];}
   if(weight>0){double s=hph+pars[5]/weight,c0=amm+amd,c1=amd+add,km=c0/s,kd=c1/s;
    double innovation=weighted_y/weight-latent;mu[p]=pm+km*innovation;dev[p]=pd+kd*innovation;
    pmm[p]=amm-km*c0;pmd[p]=amd-km*c1;pdd[p]=add-kd*c1;
    seen_exposure[p]+=weight;observed[p]=day[i];}
   else{mu[p]=pm;dev[p]=pd;pmm[p]=amm;pmd[p]=amd;pdd[p]=add;}
   seen[p]=true;last[p]=day[i];prevseason[p]=season[source];
  }
  i=end;
 }
}
