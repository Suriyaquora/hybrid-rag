"""Synthesises the 3-minute hook score (numpy only) -> hook_music.wav
Mohanam-flavoured pentatonic (D E F# A B), synced to the shot list in hook.html."""
import numpy as np, wave, sys
SR=44100; DUR=180; N=SR*DUR
rng=np.random.default_rng(7)
L=np.zeros(N,np.float32); R=np.zeros(N,np.float32)
WET_L=np.zeros(N,np.float32); WET_R=np.zeros(N,np.float32)
D4=293.66
def f(semi,base=D4): return base*2**(semi/12)
def add(buf,start,sig,gain=1.0):
    i=int(start*SR)
    if i>=N: return
    j=min(N,i+len(sig)); buf[i:j]+=sig[:j-i]*gain
def put(start,sig,gain=1.0,pan=0.0,wet=0.0):
    gl=gain*np.cos((pan+1)*np.pi/4); gr=gain*np.sin((pan+1)*np.pi/4)
    add(L,start,sig,gl); add(R,start,sig,gr)
    if wet: add(WET_L,start,sig,gl*wet); add(WET_R,start,sig,gr*wet)
def tt(d): return np.arange(int(d*SR))/SR
def lp(x,k):  # moving-average lowpass, k samples, 2 passes
    for _ in range(2):
        c=np.cumsum(np.insert(x,0,0)); x=(c[k:]-c[:-k])/k; x=np.concatenate([x,np.zeros(k-1)])
    return x
def noise(d): return rng.standard_normal(int(d*SR)).astype(np.float32)
# ---------- instruments ----------
def drum(freq=95,d=.55,click=.35):
    t=tt(d); pitch=freq*(1+1.2*np.exp(-t*28)); ph=2*np.pi*np.cumsum(pitch)/SR
    s=np.sin(ph)*np.exp(-t*9); s+=click*lp(noise(d),6)*np.exp(-t*60); return s.astype(np.float32)
def taiko(d=1.1): return drum(48,d,.5)*1.3
def pluck(freq,d=3.0,bright=1.0):
    t=tt(d); s=np.zeros_like(t)
    for n in range(1,10): s+=np.sin(2*np.pi*freq*n*(1+.0006*n*n)*t)/n**1.1*np.exp(-t*(1.6+n*1.1/bright))
    s*=np.minimum(1,t*400); return (s*.5).astype(np.float32)
def pad(freqs,d,att=2.0,rel=2.0,vib=.0):
    t=tt(d); s=np.zeros_like(t)
    for fr in freqs:
        for det in (-.004,0,.004):
            s+=np.sin(2*np.pi*fr*(1+det)*t+ (vib*np.sin(2*np.pi*5*t)))+0.35*np.sin(2*np.pi*2*fr*(1+det)*t)
    env=np.minimum(1,t/att)*np.minimum(1,(d-t)/rel); return (s*env/ (len(freqs)*3)).astype(np.float32)
def thunder(d=3.2): t=tt(d); return (lp(noise(d),90)*np.exp(-t*1.1)*np.minimum(1,t*30)*3.2).astype(np.float32)
def wind(d,amt=1): t=tt(d); return (lp(noise(d),40)*(.6+.4*np.sin(2*np.pi*t/7))*np.minimum(1,t/2)*np.minimum(1,(d-t)/2)*amt*1.6).astype(np.float32)
def riser(d,lo=200,hi=2400):
    t=tt(d); fr=lo*(hi/lo)**(t/d); ph=2*np.pi*np.cumsum(fr)/SR
    return ((np.sin(ph)*.4+lp(noise(d),4)*.7*(t/d))*(t/d)**2).astype(np.float32)
def boom(d=4.5):
    t=tt(d); fr=38+60*np.exp(-t*5); ph=2*np.pi*np.cumsum(fr)/SR
    return (np.sin(ph)*np.exp(-t*.9)+.6*lp(noise(d),30)*np.exp(-t*2.2)*4).astype(np.float32)
def reverse(sig): return sig[::-1].copy()

# ---------- A  0-12  storm + heartbeat ----------
put(0,wind(13,1.0),.55,0)
for tm in [1.5,4.0,4.4,8.0,10.5]: put(tm,thunder(),.9,rng.uniform(-.5,.5),.3)
for k in np.arange(0.5,12,1.45): put(k,drum(70,.5,.1),.55); put(k+.3,drum(60,.5,.1),.4)
put(0,pad([f(-24),f(-17)],14,3,3),.5)
put(6,riser(6,150,900),.22,0)
# ---------- B 12-30 march of empires ----------
put(11.5,boom(),.9)
for k in np.arange(12,30,.5):
    acc=1.0 if int((k-12)/.5)%4==0 else .55; put(k,taiko(),acc*.8,-.2)
    if int((k-12)/.5)%2==1: put(k+.25,drum(130,.3,.5),.35,.3)
put(12,pad([f(-24),f(-17),f(-12)],18.5,2,2),.55)
for k,s in zip(np.arange(14,29,2.0),[0,0,-3,-3,0,-5,-5,0]): put(k,pluck(f(s-12,D4),1.8,.6),.5,0)
put(24,riser(6,200,1800),.3)
# ---------- C 30-45 Parambu dawn ----------
put(29.5,boom(3),.5)
put(30,pad([f(-12),f(-5),f(0),f(2)],16,3,3),.6,0,.4)
mel=[9,7,4,7,9,12,9,7,4,2,0,2,4]
for i,s in enumerate(mel): put(31+i*1.1,pluck(f(s),3.5),.55,-.4+.8*(i%3)/2,.5)
for k in np.arange(31,45,.9): put(k,pluck(f(-12),4.0,.4),.25,0,.3)  # tanpura-ish
# ---------- D 45-70 Pari rises ----------
put(44.7,riser(1.6,300,1500),.3)
put(45,boom(4),.6)
for a,chord in [(45,[f(-12),f(-5),f(2),f(7)]),(53,[f(-10),f(-3),f(2),f(6)]),(61,[f(-12),f(-5),f(0),f(4)])]:
    put(a,pad(chord,9,2.5,2.5),.75,0,.5)
for k in np.arange(45,70,2.0): put(k,taiko(1.4),.7)
hero=[0,4,7,9,12,9,7,4, 2,4,7,9,11,9,7,2]
for i,s in enumerate(hero): put(48+i*1.4,pluck(f(s+0),3.2,1.2),.6,(-.3)+.6*(i%2),.5)
put(60,pad([f(12),f(16),f(19)],9,3,3),.25,0,.6)
# ---------- E 70-95 jasmine ----------
put(70,pad([f(-12),f(-8),f(-5),f(0)],25,3,4),.5,0,.5)
arp=[0,4,7,12,7,4,9,12,9,4,7,4]
for i in range(30): put(71+i*.8,pluck(f(arp[i%len(arp)]+(12 if i%5==0 else 0)),2.6),.42,-.5+(i%4)/3,.6)
for k in np.arange(72,95,3.0): put(k,drum(110,.6,.05),.3)
# ---------- F 95-120 poet & chieftain (night, sparse) ----------
put(95,pad([f(-12),f(-5),f(4)],25,4,4),.45,0,.7)
for i,s in enumerate([9,7,4,0,2,4,2,0,-3,0,2,4,7,4,2,0]): put(96+i*1.5,pluck(f(s),4.2,.8),.55,-.3+.6*(i%2),.8)
for tm in [100,108,115]: put(tm,reverse(riser(1.2,500,3000)),.12,0.5,.5)
# ---------- G 120-140 ultimatum: tension ----------
put(120,pad([f(-24),f(-19)],21,2,3),.6)
for k in np.arange(122,140,1.0):
    put(k,drum(75,.5,.2),.55*min(1,(k-120)/8+.3)); 
for k in np.arange(130,140,.5): put(k,taiko(),.7*(k-130)/10+.3)
put(131,riser(9,120,1800),.4)
put(137,pad([f(-12),f(-5),f(2)],4,.5,1),.4)
# ---------- H 140-160 siege: relentless ----------
put(139.7,boom(),1.0)
pat=[1,0,.5,.6, 1,0,.5,.6]
for k in np.arange(140,160,.25):
    i=int((k-140)/.25)%8
    if pat[i]: put(k,taiko(.8) if pat[i]==1 else drum(120,.3,.4),pat[i]*.8,(-.3 if i%2 else .3))
for tm in [148.2,152.6,156.1]: put(tm,thunder(),1.0,0,.3)
for k in np.arange(140,160,2.0): put(k,pad([f(-24),f(-19),f(-12)],1.9,.05,.8),.55)
for i in range(20): put(140+i,pluck(f([0,2,4,7,4,2][i%6]+12),1.2,2.0),.4,(-.5+i%5*.25),.3)
put(154,riser(6,200,3000),.35)
# ---------- I 160-170 hush ----------
put(160.2,pluck(f(12),6.0),.55,0,1.0)
put(162,pad([f(-5),f(7)],8,3,3),.25,0,.8)
put(164.5,pluck(f(9),6.0),.35,.3,1.0)
put(166.5,pluck(f(4),6.0),.30,-.3,1.0)
put(166,riser(4.4,250,3200),.3)
put(169.6,np.zeros(10,np.float32))
# ---------- J 170-180 title ----------
put(169.95,boom(5.5),1.1)
put(170,taiko(2.0),1.0)
put(170,pad([f(-24),f(-12),f(-5),f(0),f(4)],10,.1,3),.9,0,.5)
put(170.5,pad([f(12),f(16),f(19)],9,1.5,3),.4,0,.7)
for i,s in enumerate([0,4,7,12,16,12]): put(171+i*.9,pluck(f(s+12),4),.55,-.4+.16*i,.8)
for k in [172,174,176]: put(k,taiko(1.6),.8)
# ---------- reverb & master ----------
ir_t=tt(2.4); ir=(rng.standard_normal(len(ir_t))*np.exp(-ir_t*2.1)).astype(np.float32); ir[:200]*=np.linspace(0,1,200); ir/=np.sqrt((ir**2).sum())
def conv(x):
    n=1<<int(np.ceil(np.log2(len(x)+len(ir)))); return np.fft.irfft(np.fft.rfft(x,n)*np.fft.rfft(ir,n),n)[:len(x)].astype(np.float32)
L+=conv(WET_L)*1.0; R+=conv(WET_R)*1.0
# gentle compressor-ish normalisation + fades
env=np.ones(N,np.float32); env[:SR*2]=np.linspace(0,1,SR*2); env[-int(1.2*SR):]=np.linspace(1,0,int(1.2*SR))
L*=env; R*=env
peak=max(np.abs(L).max(),np.abs(R).max()); g=.88/peak
out=np.stack([np.tanh(L*g*1.15),np.tanh(R*g*1.15)],1)
out=(out/np.abs(out).max()*.89*32767).astype(np.int16)
with wave.open(sys.argv[1] if len(sys.argv)>1 else "hook_music.wav","wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(out.tobytes())
print("wrote", len(out)/SR, "s")
