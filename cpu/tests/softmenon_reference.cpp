// Independent scalar interpretation of the complete SoftMenon pipeline.
// The oracle uses coordinate-based RAW reflection, signed integer division,
// and plain sorting; it does not call the production interpolation stages.
#include "cpu_debayer.hpp"
#include "chroma_median.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool okay,const std::string& label) {if(!okay)throw std::runtime_error(label);}
int clipped(int value) {return std::max(0,std::min(255,value));}
int reflected(int coordinate,int length) {
    while(coordinate<0 || coordinate>=length)
        coordinate=coordinate<0 ? -coordinate : 2*(length-1)-coordinate;
    return coordinate;
}
int64_t rounded(int64_t numerator,int64_t denominator) {
    numerator+=denominator/2;
    return numerator>=0 ? numerator/denominator : -((-numerator+denominator-1)/denominator);
}
int measured(int x,int y,bool rggb) {
    if((x^y)&1)return 1;
    return ((x&1)==0)==rggb ? 2 : 0;
}
struct Buffer {
    static constexpr size_t prefix=29,suffix=37;
    int width,height,channels,pitch;
    std::vector<uint8_t> bytes;
    Buffer(int w,int h,int c,int padding):width(w),height(h),channels(c),pitch(w*c+padding),
        bytes(prefix+static_cast<size_t>(pitch)*h+suffix,173) {}
    uint8_t* data(){return bytes.data()+prefix;}
    const uint8_t* data()const{return bytes.data()+prefix;}
    uint8_t& at(int x,int y,int c=0){return data()[static_cast<size_t>(y)*pitch+x*channels+c];}
    int at(int x,int y,int c=0)const{return data()[static_cast<size_t>(y)*pitch+x*channels+c];}
    void reset(){std::fill(bytes.begin(),bytes.end(),173);}
    void guards()const{
        for(size_t i=0;i<prefix;++i)require(bytes[i]==173,"prefix overwritten");
        for(size_t i=prefix+static_cast<size_t>(pitch)*height;i<bytes.size();++i)require(bytes[i]==173,"suffix overwritten");
        for(int y=0;y<height;++y)for(int x=width*channels;x<pitch;++x)
            require(data()[static_cast<size_t>(y)*pitch+x]==173,"row padding overwritten");
    }
};

void completed_reference(const Buffer& raw,Buffer& rgb,bool rggb) {
    const auto sample=[&](int x,int y) {
        return raw.at(reflected(x,raw.width),reflected(y,raw.height));
    };
    const auto candidate=[&](int x,int y,int dx,int dy) {
        return rounded(sample(x-dx,y-dy)+sample(x+dx,y+dy),2)
            +rounded(6*(2*sample(x,y)-sample(x-2*dx,y-2*dy)-sample(x+2*dx,y+2*dy)),32);
    };
    const auto difference=[&](int x,int y,int dx,int dy) {
        return sample(x,y)-candidate(x,y,dx,dy);
    };
    const auto variation=[&](int x,int y,int dx,int dy) {
        return std::abs(difference(x,y,dx,dy)-difference(x-2*dx,y-2*dy,dx,dy))
            +std::abs(difference(x,y,dx,dy)-difference(x+2*dx,y+2*dy,dx,dy));
    };
    const auto score=[&](int x,int y,int dx,int dy) {
        const int bx=dy,by=dx;
        return 3*variation(x,y,dx,dy)
            +variation(x-2*bx,y-2*by,dx,dy)+variation(x+2*bx,y+2*by,dx,dy)
            +std::abs(difference(x-dx-bx,y-dy-by,dx,dy)-difference(x+dx-bx,y+dy-by,dx,dy))
            +std::abs(difference(x-dx+bx,y-dy+by,dx,dy)-difference(x+dx+bx,y+dy+by,dx,dy));
    };
    // Evaluate the stencil at virtual border coordinates. Reflecting a
    // completed green image instead would change the edge arithmetic.
    Buffer greens(raw.width+2,raw.height+2,1,0);
    for(int y=-1;y<=raw.height;++y)for(int x=-1;x<=raw.width;++x) {
        int green=sample(x,y);
        if(((x^y)&1)==0) {
            const int64_t sh=score(x,y,1,0)+1,sv=score(x,y,0,1)+1;
            green=clipped(static_cast<int>(rounded(
                sv*sv*candidate(x,y,1,0)+sh*sh*candidate(x,y,0,1),sv*sv+sh*sh)));
        }
        greens.at(x+1,y+1)=static_cast<uint8_t>(green);
    }
    const auto green=[&](int x,int y){return greens.at(x+1,y+1);};
    const auto color_difference=[&](int x,int y){return sample(x,y)-green(x,y);};
    for(int y=0;y<raw.height;++y)for(int x=0;x<raw.width;++x) {
        rgb.at(x,y,1)=static_cast<uint8_t>(green(x,y));
        const int observed=measured(x,y,rggb);
        for(int channel:{0,2}) {
            if(observed==channel){rgb.at(x,y,channel)=static_cast<uint8_t>(sample(x,y));continue;}
            int64_t d;
            if(observed==1) {
                const bool horizontal=measured(x-1,y,rggb)==channel;
                const int dx=horizontal?1:0,dy=horizontal?0:1;
                d=rounded(color_difference(x-dx,y-dy)+color_difference(x+dx,y+dy),2);
            } else {
                const int ul=color_difference(x-1,y-1),ur=color_difference(x+1,y-1);
                const int dl=color_difference(x-1,y+1),dr=color_difference(x+1,y+1);
                const int first_variation=std::abs(ul-dr),second_variation=std::abs(ur-dl);
                const int64_t first=rounded(ul+dr,2),second=rounded(ur+dl,2);
                const int64_t best=first_variation<=second_variation ? first : second;
                const int gap=std::abs(first_variation-second_variation);
                d=gap<=16 ? rounded(first+second,2) : gap<=64 ? rounded(2*best+first+second,4) : best;
            }
            rgb.at(x,y,channel)=static_cast<uint8_t>(clipped(static_cast<int>(green(x,y)+d)));
        }
    }
}

std::vector<uint8_t> reference(const Buffer& source,bool rggb,size_t& underflow,size_t& overflow) {
    std::vector<uint8_t> result(static_cast<size_t>(source.width)*source.height*3);
    for(int y=0;y<source.height;++y)for(int x=0;x<source.width;++x) {
        int medians[2];
        for(int channel:{0,2}) {
            std::array<int,9> differences;int index=0;
            for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
                const int xx=reflected(x+dx,source.width),yy=reflected(y+dy,source.height);
                differences[index++]=source.at(xx,yy,channel)-source.at(xx,yy,1);
            }
            std::sort(differences.begin(),differences.end());
            medians[channel/2]=differences[4];
        }
        const int channel=measured(x,y,rggb);
        const int unbounded_green=channel==1 ? source.at(x,y,1) : source.at(x,y,channel)-medians[channel/2];
        underflow+=unbounded_green<0;overflow+=unbounded_green>255;
        const int green=clipped(unbounded_green);
        const size_t offset=(static_cast<size_t>(y)*source.width+x)*3;
        result[offset+1]=static_cast<uint8_t>(green);
        for(int c:{0,2})result[offset+c]=static_cast<uint8_t>(c==channel ? source.at(x,y,c) : clipped(green+medians[c/2]));
    }
    return result;
}

void compare(const Buffer& actual,const std::vector<uint8_t>& expected,bool swap,const std::string& label) {
    actual.guards();
    for(int y=0;y<actual.height;++y)for(int x=0;x<actual.width;++x)for(int c=0;c<3;++c) {
        const int target=swap ? 2-c : c;
        const int value=expected[(static_cast<size_t>(y)*actual.width+x)*3+target];
        if(actual.at(x,y,c)!=value)throw std::runtime_error(label+" mismatch at "+std::to_string(x)+","+std::to_string(y)+","+std::to_string(c));
    }
}
}

int main() {
    try {
        std::mt19937 rng(20261010);
        std::array<std::unique_ptr<Debayer>,3> backends;
        const int workers[]={1,3,7};for(int i=0;i<3;++i)backends[i]=std::make_unique<Debayer>(workers[i]);
        using Cleanup=void(*)(const uint8_t*,size_t,uint8_t*,size_t,int,int,bool,int,int,bool);
        std::vector<std::pair<std::string,Cleanup>> cleanups{
            {"insertion",median_cpu_detail::scalar_rows<true>},
            {"network",median_cpu_detail::scalar_rows<false>}};
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(__builtin_cpu_supports("avx2"))cleanups.emplace_back("AVX2",median_cpu_detail::avx2_rows);
        if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512bw")&&__builtin_cpu_supports("avx512vbmi"))
            cleanups.emplace_back("AVX512",median_cpu_detail::avx512_rows);
#endif
        size_t public_count=0,direct_count=0,underflow=0,overflow=0;
        const int shapes[][2]={{2,2},{2,3},{3,2},{3,3},{7,9},{9,7},{15,17},{16,16},{17,15},
            {31,33},{32,32},{33,31},{63,65},{64,64},{65,63},{127,129},{129,127},{193,97},{257,129},{1921,129}};
        for(const auto& shape:shapes)for(bool rggb:{false,true})for(int kind=0;kind<5;++kind) {
            const int w=shape[0],h=shape[1];
            Buffer raw(w,h,1,11),pre(w,h,3,17),out(w,h,3,23);
            for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
                uint8_t value=static_cast<uint8_t>(rng()>>24);
                if(kind==0)value=0;
                if(kind==1)value=255;
                if(kind==2)value=value>127?255:0;
                if(kind==3)value=std::array<uint8_t,3>{17,101,239}[measured(x,y,rggb)];
                raw.at(x,y)=value;
            }
            const auto raw_before=raw.bytes;
            raw_image_t input{w,h,raw.data(),raw.pitch,SARONIC_DEBAYER_SOFTMENON,rggb?SARONIC_DEBAYER_RGGB:SARONIC_DEBAYER_BGGR};
            bgr_image_t destination;
            completed_reference(raw,pre,rggb);
            pre.guards();const auto expected=reference(pre,rggb,underflow,overflow);
            for(auto& backend:backends) {
                out.reset();destination={w,h,out.data(),out.pitch};
                require(backend->Process(&input,&destination)==0,"public Process failed");
                compare(out,expected,false,"public");require(raw.bytes==raw_before,"input modified");++public_count;
                for(int y=0;y<h;++y)for(int x=0;x<w;++x)
                    require(out.at(x,y,measured(x,y,rggb))==raw.at(x,y),"measured CFA changed");
            }
            // Arbitrary RGB sources exercise median/green clipping cases that
            // a demosaicer may rarely produce, as well as independent pitches.
            if(kind==4)for(int y=0;y<h;++y)for(int x=0;x<w;++x)for(int c=0;c<3;++c)
                pre.at(x,y,c)=static_cast<uint8_t>(rng()>>24);
            const auto direct_expected=reference(pre,rggb,underflow,overflow),pre_before=pre.bytes;
            std::vector<int> splits{0,1,h/3,h/2+1,h-1,h};
            std::sort(splits.begin(),splits.end());splits.erase(std::unique(splits.begin(),splits.end()),splits.end());
            for(const auto& cleanup:cleanups)for(bool swap:{false,true}) {
                out.reset();
                for(size_t i=splits.size()-1;i>0;--i) {
                    const int begin=splits[i-1],end=splits[i];const auto before=out.bytes;
                    cleanup.second(pre.data(),pre.pitch,out.data(),out.pitch,w,h,rggb,begin,end,swap);
                    for(int y=0;y<h;++y)if(y<begin||y>=end) {
                        const size_t offset=Buffer::prefix+static_cast<size_t>(y)*out.pitch;
                        require(std::equal(out.bytes.begin()+offset,out.bytes.begin()+offset+out.pitch,before.begin()+offset),"cleanup wrote outside slice");
                    }
                }
                compare(out,direct_expected,swap,cleanup.first);require(pre.bytes==pre_before,"cleanup modified source");++direct_count;
            }
        }
        require(underflow>0&&overflow>0,"green clipping cases were not exercised");
        std::cout<<"SoftMenon independent reference: "<<public_count<<" public outputs, "<<direct_count
                 <<" direct sliced outputs, "<<underflow<<" green underflows, "<<overflow<<" green overflows\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"SoftMenon reference failure: "<<error.what()<<'\n';return 1;}
}
