// Scalar equivalent of run_filter_v2; caller supplies chronological player rows.
#include <cmath>
#include <cstdint>
#include <vector>
extern "C" void monthly_ou(long n, long np, const int64_t* ids,
    const double* dt, const double* sc, const double* expo, const double* obs,
    const double* pars, double mean, const double* init, const double* shift,
    const double* trans, const double* context, double lambda,
    double* pred, double* predvar, double* post, double* postvar) {
  std::vector<double> mu(np,0),d(np,0),pmm(np,0),pmd(np,0),pdd(np,0);
  std::vector<bool> seen(np,false);
  for(long i=0;i<n;i++) {
    long p=ids[i]; double pm,pd,amm,amd,add;
    if(!seen[p]) {pm=init[i];pd=0;amm=pars[1];amd=0;add=pars[3];seen[p]=true;}
    else {
      double am=std::exp(-dt[i]/pars[0]),ad=std::exp(-dt[i]/pars[2]);
      double anchor=mean+shift[i]*sc[i];
      pm=anchor+am*(mu[p]-anchor);pd=ad*d[p];
      amm=am*am*pmm[p]+pars[1]*(1-am*am)+pars[4]*sc[i];
      amd=am*ad*pmd[p];add=ad*ad*pdd[p]+pars[3]*(1-ad*ad);
    }
    double prediction=pm+pd+trans[i]+lambda*context[i];
    double hph=amm+2*amd+add,s=hph+pars[5]/expo[i];
    double c0=amm+amd,c1=amd+add,km=c0/s,kd=c1/s,innovation=obs[i]-prediction;
    mu[p]=pm+km*innovation;d[p]=pd+kd*innovation;
    pmm[p]=amm-km*c0;pmd[p]=amd-km*c1;pdd[p]=add-kd*c1;
    pred[i]=prediction;predvar[i]=hph;post[i]=mu[p]+d[p];
    postvar[i]=pmm[p]+2*pmd[p]+pdd[p];
  }
}
