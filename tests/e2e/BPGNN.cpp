#include <algorithm>
#include <backend/FSS_transformer.h>
#include <FSS/api.h>
#include <FSS/comms.h>
#include <FSS/config.h>
#include <FSS/prng.h>
#include <FSS/stats.h>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

struct Dataset { const char *name; int nodes, edges, input_dim, classes; };
struct Cost { std::string name; double total_ms, keyread_ms, online_ms; uint64_t online, offline; };
struct QueryResult {
    std::vector<Cost> layers;
    double total_ms=0, keyread_ms=0, online_ms=0;
    uint64_t online=0, offline=0;
};

static uint64_t online_comm() {
    return FSSConfig::party == DEALER ? 0 :
        FSSConfig::peer->bytesSent() + FSSConfig::peer->bytesReceived();
}
static uint64_t offline_comm() {
    return FSSConfig::party == DEALER ?
        FSSConfig::server->bytesSent() + FSSConfig::client->bytesSent() :
        FSSConfig::dealer->bytesReceived();
}
static uint64_t keyread_us() {
    uint64_t total=0;for(const auto &entry:FSS::stats)total+=entry.second.keyread_time;return total;
}
template<class Fn> static Cost measure(const std::string &name, Fn &&fn) {
    uint64_t on=online_comm(),off=offline_comm(),key0=keyread_us();auto t=Clock::now();fn();
    uint64_t total_us=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-t).count();
    uint64_t key_us=keyread_us()-key0;
    return {name,total_us/1000.0,key_us/1000.0,
            (total_us>=key_us?total_us-key_us:0)/1000.0,
            online_comm()-on,offline_comm()-off};
}
static void add(QueryResult &q, Cost c) {q.total_ms+=c.total_ms;q.keyread_ms+=c.keyread_ms;
    q.online_ms+=c.online_ms;q.online+=c.online;q.offline+=c.offline;q.layers.push_back(std::move(c));}

static GroupElement mask_at(uint64_t i,uint64_t salt){uint64_t z=i+salt+0x9e3779b97f4a7c15ULL;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static void fill_masked(std::vector<GroupElement>&x,std::vector<GroupElement>&m,uint64_t salt,uint64_t modv){
    for(size_t i=0;i<x.size();++i){auto r=mask_at(i,salt),v=GroupElement((i*7+salt)%modv);
        if(FSSConfig::party==DEALER)m[i]=r;else x[i]=v+r;}}
static void fill_one(std::vector<GroupElement>&x,std::vector<GroupElement>&m,uint64_t salt){
    for(size_t i=0;i<x.size();++i){auto r=mask_at(i,salt);if(FSSConfig::party==DEALER)m[i]=r;else x[i]=1+r;}}

static GraphitiGraph make_graph(int n,int edges,int salt=0){
    std::vector<int>s(edges),d(edges);for(int e=0;e<edges;++e){s[e]=e%n;
        d[e]=(e<n)?s[e]:(s[e]*17+e/n+1+salt)%n;}return GraphitiGraph(n,std::move(s),std::move(d));}
static GraphitiGraph make_delta_graph(int nd,int ng,int query){
    std::vector<int>s,d;for(int i=0;i<nd;++i){s.push_back(i);d.push_back(i);
        if(ng){s.push_back(nd+(i+query)%ng);d.push_back(i);}}
    for(int q=0;q<ng;++q){s.push_back((q+query)%nd);d.push_back(nd+q);}
    return GraphitiGraph(nd+ng,std::move(s),std::move(d));}

static void print_query(const Dataset &ds,int round,const QueryResult&q){
    if(FSSConfig::party!=CLIENT)return;std::cout<<"\n["<<ds.name<<"] Round "<<round
      <<(round==1?" - STATIC (base preprocessing generated)":" - DYNAMIC (base preprocessing reused)")<<'\n';
    std::cout<<std::left<<std::setw(29)<<"Stage"<<std::right<<std::setw(13)<<"Total(ms)"
      <<std::setw(14)<<"KeyRead(ms)"<<std::setw(14)<<"Online(ms)"
      <<std::setw(16)<<"Online(MB)"<<std::setw(17)<<"Offline(MB)\n"<<std::string(103,'-')<<'\n';
    for(auto&c:q.layers)std::cout<<std::left<<std::setw(29)<<c.name<<std::right<<std::fixed<<std::setprecision(3)
      <<std::setw(13)<<c.total_ms<<std::setw(14)<<c.keyread_ms<<std::setw(14)<<c.online_ms
      <<std::setprecision(6)<<std::setw(16)<<c.online/1048576.0
      <<std::setw(17)<<c.offline/1048576.0<<'\n';
    std::cout<<std::string(103,'-')<<"\nTOTAL"<<std::setw(32)<<std::fixed<<std::setprecision(3)<<q.total_ms
      <<std::setw(14)<<q.keyread_ms<<std::setw(14)<<q.online_ms
      <<std::setw(16)<<std::setprecision(6)<<q.online/1048576.0<<std::setw(17)<<q.offline/1048576.0<<'\n';
}
static void print_summary(const Dataset&ds,const std::vector<QueryResult>&r){
    if(FSSConfig::party!=CLIENT)return;double dt=0,dk=0,dol=0,don=0,doff=0;
    for(int i=1;i<6;++i){dt+=r[i].total_ms;dk+=r[i].keyread_ms;dol+=r[i].online_ms;don+=r[i].online;doff+=r[i].offline;}
    dt/=5;dk/=5;dol/=5;don/=5;doff/=5;std::cout<<"\n================ "<<ds.name<<" SUMMARY ================\n"
      <<std::left<<std::setw(24)<<"Setting"<<std::right<<std::setw(14)<<"Total(ms)"<<std::setw(14)<<"KeyRead(ms)"
      <<std::setw(14)<<"Online(ms)"<<std::setw(18)<<"Online(MB)"<<std::setw(18)<<"Offline(MB)\n"<<std::string(102,'-')<<'\n'
      <<std::left<<std::setw(24)<<"Static (round 1)"<<std::right<<std::fixed<<std::setprecision(3)
      <<std::setw(14)<<r[0].total_ms<<std::setw(14)<<r[0].keyread_ms<<std::setw(14)<<r[0].online_ms
      <<std::setprecision(6)<<std::setw(18)<<r[0].online/1048576.0<<std::setw(18)<<r[0].offline/1048576.0<<'\n'
      <<std::left<<std::setw(24)<<"Dynamic avg (rounds 2-6)"<<std::right<<std::setprecision(3)<<std::setw(14)<<dt
      <<std::setw(14)<<dk<<std::setw(14)<<dol<<std::setprecision(6)<<std::setw(18)<<don/1048576.0<<std::setw(18)<<doff/1048576.0<<'\n'
      <<std::left<<std::setw(24)<<"Static / Dynamic"<<std::right<<std::setprecision(3)<<std::setw(14)<<r[0].total_ms/dt
      <<std::setw(14)<<(dk?r[0].keyread_ms/dk:0)<<std::setw(14)<<(dol?r[0].online_ms/dol:0)
      <<std::setw(18)<<(don?r[0].online/don:0)<<std::setw(18)<<(doff?r[0].offline/doff:0)<<"\n==========================================================\n";
}

static void benchmark_dataset(const Dataset&ds,bool use_grotto_relu){
    constexpr int hidden=64,delta_nodes=10,ghosts=5,rounds=6,scale=12;
    GraphitiGraph base_plain=make_graph(ds.nodes,ds.edges),base_plan;
    std::vector<GroupElement>w1(size_t(ds.input_dim)*hidden),w1m(w1.size()),w2(size_t(hidden)*ds.classes),w2m(w2.size());
    fill_masked(w1,w1m,1001+ds.nodes,3);fill_masked(w2,w2m,2003+ds.nodes,3);
    std::vector<GroupElement>base_x(size_t(ds.nodes)*ds.input_dim),base_xm(base_x.size());fill_masked(base_x,base_xm,3001+ds.nodes,5);
    std::vector<GroupElement>cached_h1(size_t(ds.nodes)*hidden),cached_h1m(cached_h1.size());
    std::vector<QueryResult>results(rounds);

    for(int round=1;round<=rounds;++round){bool dynamic=round>1;int nd=dynamic?delta_nodes:0,ng=dynamic?ghosts:0,nt=ds.nodes+nd;
        GraphitiGraph delta_plain=dynamic?make_delta_graph(nd,ng,round):GraphitiGraph(0,{},{}),delta_plan;
        std::vector<int>ghost_to_base(ng);for(int q=0;q<ng;++q)ghost_to_base[q]=(q*997+round)%ds.nodes;
        std::vector<GroupElement>degree(nt),degreem(nt);fill_one(degree,degreem,4001+round+ds.nodes);
        std::vector<GroupElement>h1(size_t(nt)*hidden),h1m(h1.size()),a1(h1.size()),a1m(h1.size()),r1(h1.size()),r1m(h1.size());
        std::vector<GroupElement>h2(size_t(nt)*ds.classes),h2m(h2.size()),a2(h2.size()),a2m(h2.size()),y(h2.size()),ym(h2.size());
        FSS::start();
        if(round==1)add(results[round-1],measure("Graphiti base preprocessing",[&]{base_plan=GraphitiPrepare2P1(base_plain);}));
        else add(results[round-1],measure("Graphiti base preprocessing",[&]{base_plan.validate();}));
        if(dynamic)add(results[round-1],measure("Graphiti delta preprocessing",[&]{delta_plan=GraphitiPrepare2P1(delta_plain);}));
        else delta_plan=delta_plain;

        if(!dynamic){add(results[0],measure("Layer 1: base MatMul",[&]{MatMul2D(ds.nodes,ds.input_dim,hidden,base_x.data(),base_xm.data(),w1.data(),w1m.data(),h1.data(),h1m.data(),true);}));
            cached_h1=h1;cached_h1m=h1m;}
        else{std::copy(cached_h1.begin(),cached_h1.end(),h1.begin());std::copy(cached_h1m.begin(),cached_h1m.end(),h1m.begin());
            std::vector<GroupElement>dx(size_t(nd)*ds.input_dim),dxm(dx.size()),dh(size_t(nd)*hidden),dhm(dh.size());fill_masked(dx,dxm,5003+round+ds.nodes,5);
            add(results[round-1],measure("Layer 1: delta MatMul",[&]{MatMul2D(nd,ds.input_dim,hidden,dx.data(),dxm.data(),w1.data(),w1m.data(),dh.data(),dhm.data(),true);}));
            std::copy(dh.begin(),dh.end(),h1.begin()+size_t(ds.nodes)*hidden);std::copy(dhm.begin(),dhm.end(),h1m.begin()+size_t(ds.nodes)*hidden);}
        add(results[round-1],measure("Layer 1: BPMPL",[&]{BPMPLGraphitiRouting(base_plan,delta_plan,nd,ng,hidden,h1.data(),h1m.data(),degree.data(),degreem.data(),a1.data(),a1m.data(),ghost_to_base.data(),"L1::");}));
        if(use_grotto_relu)
            add(results[round-1],measure("Layer 1: Grotto-ReLU",[&]{GrottoReLU(nt*hidden,a1.data(),r1.data(),a1m.data(),r1m.data(),"L1::");}));
        else
            add(results[round-1],measure("Layer 1: GTDCF-ReLU",[&]{GTDCFReLU(nt*hidden,a1.data(),r1.data(),a1m.data(),r1m.data(),8,"L1::");}));
        add(results[round-1],measure("Layer 2: MatMul",[&]{MatMul2D(nt,hidden,ds.classes,r1.data(),r1m.data(),w2.data(),w2m.data(),h2.data(),h2m.data(),true);}));
        add(results[round-1],measure("Layer 2: BPMPL",[&]{BPMPLGraphitiRouting(base_plan,delta_plan,nd,ng,ds.classes,h2.data(),h2m.data(),degree.data(),degreem.data(),a2.data(),a2m.data(),ghost_to_base.data(),"L2::");}));
        add(results[round-1],measure("Output: Softmax",[&]{BPGCNSoftmax(nt,ds.classes,a2.data(),y.data(),a2m.data(),ym.data(),scale,"Output::");}));
        FSS::end();print_query(ds,round,results[round-1]);
    }print_summary(ds,results);
}

int main(int argc,char**argv){sytorch_init();if(argc<2){std::cerr<<"Usage: "<<argv[0]<<" <party:1|2|3> [ip] [--relu=gtdcf|--relu=grotto]\n";return 1;}
    FSSConfig::party=std::atoi(argv[1]);FSSConfig::bitlength=64;FSSConfig::num_threads=4;
    std::string ip="127.0.0.1";bool use_grotto_relu=false;
    for(int i=2;i<argc;++i){std::string arg=argv[i];
        if(arg=="--relu=grotto")use_grotto_relu=true;
        else if(arg=="--relu=gtdcf")use_grotto_relu=false;
        else ip=arg;}
    for(int i=0;i<256;++i)FSSConfig::prngs[i].SetSeed(osuCrypto::toBlock(0x12345678ULL+i,0xabcdefULL));
    // Stream Dealer keys from server.dat/client.dat. Loading an entire key file
    // (memBuf=true) can exhaust RAM for Pubmed and be killed by the OS.
    FSSTransformer<u64>backend;backend.init(ip,false);
#if defined(BPGNN_CORA_ONLY)
    const Dataset datasets[]={{"Cora",2708,5429,1433,7}};
#elif defined(BPGNN_CITESEER_ONLY)
    const Dataset datasets[]={{"Citeseer",3327,4732,3703,6}};
#elif defined(BPGNN_PUBMED_ONLY)
    const Dataset datasets[]={{"Pubmed",19717,44338,500,3}};
#else
    const Dataset datasets[]={{"Cora",2708,5429,1433,7},{"Citeseer",3327,4732,3703,6},{"Pubmed",19717,44338,500,3}};
#endif
    for(const auto&ds:datasets)benchmark_dataset(ds,use_grotto_relu);backend.finalize();return 0;}
