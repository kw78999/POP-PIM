// Host (S-path) Top-k timing: R rows of the [32,N] BF16 score replay, 8 threads (spawn/join timed),
// nth_element + final sort + index sort; stable BF16 order: score desc, index asc, +-0 equal.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
using namespace std; using Clock = chrono::steady_clock;
static uint16_t key(uint16_t x){if((x&32767)==0)x=0;return x&32768?uint16_t(~x):x^32768;}
struct Compare{const uint16_t*s;bool operator()(int a,int b)const{auto x=key(s[a]),y=key(s[b]);return x!=y?x>y:a<b;}};
static double quant(vector<double>x,double p){sort(x.begin(),x.end());double a=(x.size()-1)*p;return x[(int)a]*(1-a+int(a))+x[(int)ceil(a)]*(a-int(a));}
// rows: which rows of the [32,N] score file to select from (stride 4 for GQA-shared, 1 for per-head)
static vector<int> run(const vector<uint16_t>&s,int N,int K,int R,int stride){
 vector<int>out(size_t(R)*K);vector<thread>th;
 for(int t=0;t<8;t++)th.emplace_back([&,t](){vector<int>ids(N);for(int r=t;r<R;r+=8){int h=r*stride;iota(ids.begin(),ids.end(),0);Compare c{s.data()+size_t(h)*N};nth_element(ids.begin(),ids.begin()+K,ids.end(),c);sort(ids.begin(),ids.begin()+K,c);sort(ids.begin(),ids.begin()+K);copy(ids.begin(),ids.begin()+K,out.begin()+size_t(r)*K);}});
 for(auto&x:th)x.join();return out;}
int main(int argc,char**argv){
 if(argc!=6){cerr<<"scores expected N K stride\n";return 2;}
 int N=stoi(argv[3]),K=stoi(argv[4]),stride=stoi(argv[5]),R=32/stride;
 vector<uint16_t>s(size_t(32)*N);ifstream(argv[1],ios::binary).read((char*)s.data(),s.size()*2);
 vector<int>e(size_t(32)*K);ifstream(argv[2],ios::binary).read((char*)e.data(),e.size()*4);
 auto check=[&](const vector<int>&o){for(int r=0;r<R;r++)for(int i=0;i<K;i++)if(o[size_t(r)*K+i]!=e[size_t(r*stride)*K+i])throw runtime_error("membership");};
 for(int i=0;i<10;i++)check(run(s,N,K,R,stride));
 vector<double>t;for(int i=0;i<30;i++){auto a=Clock::now();auto o=run(s,N,K,R,stride);auto b=Clock::now();check(o);t.push_back(chrono::duration<double,milli>(b-a).count());}
 cout<<"{\"rows\":"<<R<<",\"N\":"<<N<<",\"K\":"<<K<<",\"threads\":8,\"warmup\":10,\"measurements\":30,\"median_ms\":"<<quant(t,.5)<<",\"p95_ms\":"<<quant(t,.95)<<",\"membership_pass\":true}"<<endl;}
