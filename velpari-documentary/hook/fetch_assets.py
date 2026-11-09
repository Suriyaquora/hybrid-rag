"""Downloads free-licence images from Wikimedia Commons (throttled) and records credits to assets/credits.json"""
import json,os,re,sys,time,urllib.request,urllib.parse,urllib.error
sys.path.insert(0,"/tmp/claude-0/-home-user-hybrid-rag/e079f4a0-e451-5cab-8b25-d80a2d6a0599/scratchpad")
from cs import api,UA
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"assets"); os.makedirs(OUT+"/raw",exist_ok=True)
WANT={ # key: commons file title
 "hills_a":"Munnar-Hills-WUS07040.jpg","hills_b":"Munnar 2024 - hill.jpg","mist_tn":"Mist Mountains Courtallam TN Jan22 R16 05490.jpg",
 "kolli_forest":"KOLLI HILLS FOREST.jpg","jasmine_a":"Jasmine (Jasminum sambac).jpg","jasmine_b":"Jasmine flower in hand.JPG",
 "milkyway_a":"Milky Way @Pushpagiri Wildlife Sanctuary.jpg","milkyway_b":"Milky way amongst mountain.jpg",
 "lightning_a":"Lightning NOAA.jpg","lightning_b":"A Vintage Lightning Storm at Kitt Peak.jpg","lightning_c":"September 4 2019 Reno Lightning Storm.jpg",
 "palm_leaf":"11th century Tevaram, 18th century palm leaf manuscript.jpg","elephant_a":"Kuzhur - Elephants.jpg","elephant_b":"Thrippunithura-Elephant-Entering-Temple-1 crop.jpg",
}
credits=json.load(open(OUT+"/credits.json")) if os.path.exists(OUT+"/credits.json") else {}
for k,title in WANT.items():
    dst=f"{OUT}/raw/{k}.jpg"
    if os.path.exists(dst): continue
    d=api({"action":"query","titles":"File:"+title,"prop":"imageinfo","iiprop":"url|size|extmetadata","iiurlwidth":2560})
    p=list(d["query"]["pages"].values())[0]; ii=p["imageinfo"][0]; m=ii["extmetadata"]
    url=(ii.get("thumburl") or ii["url"]).split("?")[0].replace("thumb.wikimedia.org","upload.wikimedia.org"); url=re.sub(r"/\d+px-","/3840px-",url)
    for i in range(6):
        try:
            data=urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":UA}),timeout=90).read();break
        except urllib.error.HTTPError as e:
            if e.code==429: time.sleep(10);continue
            raise
    import io; from PIL import Image
    im=Image.open(io.BytesIO(data)).convert("RGB")
    if im.width>2560: im=im.resize((2560,round(im.height*2560/im.width)),Image.LANCZOS)
    im.save(dst,quality=88)
    strip=lambda s:re.sub("<[^>]+>","",s or "").strip()
    credits[k]={"title":title,"artist":strip(m.get("Artist",{}).get("value")),"license":m.get("LicenseShortName",{}).get("value"),"page":ii["descriptionurl"]}
    json.dump(credits,open(OUT+"/credits.json","w"),indent=1,ensure_ascii=False)
    print("ok",k,len(data)//1024,"KB",credits[k]["license"],flush=True); time.sleep(5)
print("done")
