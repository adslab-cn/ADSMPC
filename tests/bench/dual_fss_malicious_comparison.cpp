#include <FSS/dual_fss_auth.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock=std::chrono::steady_clock;
struct Row{std::string method;int ell;std::size_t B;double keygen=0,eval=0,b2a=0,check=0;std::size_t comm=0;};
static std::uint64_t mask64(int n){return n==64?~0ULL:((1ULL<<n)-1);}
static DualFSSAuthShare tampered(DualFSSAuthShare x){x.value^=1;return x;}
static void initNativePrng(std::uint64_t seed){for(int i=0;i<256;++i)FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(seed+i,0xdeadbeefbadc0ffeULL));}

static void selfTest(int s){
 for(int ell:{32,64}){
  GroupElement d0=7,d1=11,delta=d0+d1;std::uint64_t alpha=17;
  auto check=[&](const auto& fn,const auto& predicate,const char* name){
   for(std::uint64_t x:{16ULL,17ULL,18ULL}){auto z=fn(x);auto r=batchCheckDualFSS({z.first},{z.second},d0,d1,ell,s,100+x);if(!r.accepted||r.opened.size()!=1||r.opened[0]!=GroupElement(predicate(x)))throw std::runtime_error(std::string(name)+" correctness");}
   auto z=fn(18);auto bad=tampered(z.second);if(batchCheckDualFSS({z.first},{bad},d0,d1,ell,s,999).accepted)throw std::runtime_error(std::string(name)+" tamper accepted");
  };
  auto lt=[&](std::uint64_t x){return x<alpha;};auto ge=[&](std::uint64_t x){return x>=alpha;};
  auto dk=keyGenDualFSSDCF(ell,ell,s,alpha,delta);check([&](std::uint64_t x){return std::make_pair(evalDualFSSDCF(0,dk,x),evalDualFSSDCF(1,dk,x));},lt,"Dual-FSS DCF");freeDualFSSDCF(dk);
  auto gk=keyGenDualFSSGrotto(ell,ell,s,alpha,delta,0x12345678);check([&](std::uint64_t x){DualFSSAuthShare a,b;evalDualFSSGrotto(gk,x,ell,s,a,b);return std::make_pair(a,b);},lt,"Dual-FSS GROTTO");freeDualFSSGrotto(gk);
  auto tk=keyGenDualFSSGTDCF(ell,8,ell,s,alpha,delta);check([&](std::uint64_t x){return std::make_pair(evalDualFSSGTDCF(0,tk,x),evalDualFSSGTDCF(1,tk,x));},ge,"Dual-FSS GTDCF");freeDualFSSGTDCF(tk);
 }
}

namespace bench {
Row dcf(int ell,int s,std::size_t B,std::size_t chunk,std::mt19937_64& rng){
 Row row{"Dual-FSS-DCF(SHARK-style)",ell,B};GroupElement d0=rng()&mask64(ell),d1=rng()&mask64(ell),delta=d0+d1;std::uint64_t im=mask64(ell);std::size_t done=0;
 while(done<B){std::size_t n=std::min(chunk,B-done);std::vector<DualFSSDCFDealerKeys> keys;keys.reserve(n);std::vector<std::uint64_t>x(n);auto k0=Clock::now();for(std::size_t i=0;i<n;++i){keys.push_back(keyGenDualFSSDCF(ell,ell,s,rng()&im,delta));x[i]=rng()&im;}auto k1=Clock::now();std::vector<DualFSSAuthShare>a(n),b(n);auto e0=Clock::now();for(std::size_t i=0;i<n;++i){a[i]=evalDualFSSDCF(0,keys[i],x[i]);b[i]=evalDualFSSDCF(1,keys[i],x[i]);}auto e1=Clock::now();auto c0=Clock::now();auto z=batchCheckDualFSS(a,b,d0,d1,ell,s,rng());auto c1=Clock::now();if(!z.accepted)throw std::runtime_error("Dual DCF rejected");row.keygen+=std::chrono::duration<double,std::milli>(k1-k0).count();row.eval+=std::chrono::duration<double,std::milli>(e1-e0).count();row.check+=std::chrono::duration<double,std::milli>(c1-c0).count();row.comm+=z.communication_bytes;for(auto&k:keys)freeDualFSSDCF(k);done+=n;}return row;
}
Row grotto(int ell,int s,std::size_t B,std::size_t chunk,std::mt19937_64& rng){
 Row row{"Dual-FSS-GROTTO+B2A(LightShark-style)",ell,B};GroupElement d0=rng()&mask64(ell),d1=rng()&mask64(ell),delta=d0+d1;std::uint64_t im=mask64(ell);std::size_t done=0;
 while(done<B){std::size_t n=std::min(chunk,B-done);std::vector<DualFSSGrottoDealerKeys> keys;keys.reserve(n);std::vector<std::uint64_t>x(n);auto k0=Clock::now();for(std::size_t i=0;i<n;++i){keys.push_back(keyGenDualFSSGrotto(ell,ell,s,rng()&im,delta,rng()));x[i]=rng()&im;}auto k1=Clock::now();std::vector<DualFSSAuthShare>a(n),b(n);std::size_t b2aBytes=0;std::uint64_t b2aNs=0;auto e0=Clock::now();for(std::size_t i=0;i<n;++i)evalDualFSSGrotto(keys[i],x[i],ell,s,a[i],b[i],&b2aBytes,&b2aNs);auto e1=Clock::now();auto c0=Clock::now();auto z=batchCheckDualFSS(a,b,d0,d1,ell,s,rng());auto c1=Clock::now();if(!z.accepted)throw std::runtime_error("Dual GROTTO+B2A rejected");row.keygen+=std::chrono::duration<double,std::milli>(k1-k0).count();row.eval+=std::chrono::duration<double,std::milli>(e1-e0).count();row.b2a+=b2aNs/1000000.0;row.check+=std::chrono::duration<double,std::milli>(c1-c0).count();row.comm+=z.communication_bytes+b2aBytes;for(auto&k:keys)freeDualFSSGrotto(k);done+=n;}return row;
}
Row gtdcf(int ell,int s,std::size_t B,std::size_t chunk,std::mt19937_64& rng){
 Row row{"Dual-FSS-GTDCF",ell,B};GroupElement d0=rng()&mask64(ell),d1=rng()&mask64(ell),delta=d0+d1;std::uint64_t im=mask64(ell);std::size_t done=0;
 while(done<B){std::size_t n=std::min(chunk,B-done);std::vector<DualFSSGTDCFDealerKeys> keys;keys.reserve(n);std::vector<std::uint64_t>x(n);auto k0=Clock::now();for(std::size_t i=0;i<n;++i){keys.push_back(keyGenDualFSSGTDCF(ell,8,ell,s,rng()&im,delta));x[i]=rng()&im;}auto k1=Clock::now();std::vector<DualFSSAuthShare>a(n),b(n);auto e0=Clock::now();for(std::size_t i=0;i<n;++i){a[i]=evalDualFSSGTDCF(0,keys[i],x[i]);b[i]=evalDualFSSGTDCF(1,keys[i],x[i]);}auto e1=Clock::now();auto c0=Clock::now();auto z=batchCheckDualFSS(a,b,d0,d1,ell,s,rng());auto c1=Clock::now();if(!z.accepted)throw std::runtime_error("Dual GTDCF rejected");row.keygen+=std::chrono::duration<double,std::milli>(k1-k0).count();row.eval+=std::chrono::duration<double,std::milli>(e1-e0).count();row.check+=std::chrono::duration<double,std::milli>(c1-c0).count();row.comm+=z.communication_bytes;for(auto&k:keys)freeDualFSSGTDCF(k);done+=n;}return row;
}
Row median(std::vector<Row> rows){auto md=[&](auto f){std::vector<double>v;for(auto&r:rows)v.push_back(f(r));std::sort(v.begin(),v.end());return v[v.size()/2];};Row z=rows.front();z.keygen=md([](auto&r){return r.keygen;});z.eval=md([](auto&r){return r.eval;});z.b2a=md([](auto&r){return r.b2a;});z.check=md([](auto&r){return r.check;});std::vector<std::size_t>c;for(auto&r:rows)c.push_back(r.comm);std::sort(c.begin(),c.end());z.comm=c[c.size()/2];return z;}
void print(const Row&r){std::cout<<r.method<<",Arithmetic,"<<r.ell<<','<<r.B<<','<<std::fixed<<std::setprecision(3)<<r.keygen<<','<<r.eval<<','<<r.b2a<<','<<r.check<<','<<r.eval+r.check<<','<<r.comm/1048576.0<<'\n';}
}

int main(int argc,char**argv){
 int s=argc>1?std::atoi(argv[1]):40;std::size_t maxB=argc>2?std::strtoull(argv[2],nullptr,10):100000,chunk=argc>3?std::strtoull(argv[3],nullptr,10):1000;int reps=argc>4?std::atoi(argv[4]):3;
 if(s<=0||s>64||reps<=0||!(reps&1))throw std::invalid_argument("require 1<=s<=64 and an odd positive repetition count");initNativePrng(0x4455414c465353ULL);selfTest(s);
 std::cout<<"Method,Output,ell,B,KeyGen(ms),Eval(ms),B2A(ms),BatchCheck(ms),Online(ms),Online(MB)\n";
 for(int ell:{32,64})for(std::size_t B:{std::size_t(1000),std::size_t(10000),std::size_t(100000)})if(B<=maxB){std::vector<Row>d,g,t;for(int rep=0;rep<reps;++rep){std::mt19937_64 rng(0x4455414c465353ULL^(std::uint64_t(ell)<<40)^(B<<4)^rep);d.push_back(bench::dcf(ell,s,B,chunk,rng));g.push_back(bench::grotto(ell,s,B,chunk,rng));t.push_back(bench::gtdcf(ell,s,B,chunk,rng));}bench::print(bench::median(std::move(d)));bench::print(bench::median(std::move(g)));bench::print(bench::median(std::move(t)));}
}
