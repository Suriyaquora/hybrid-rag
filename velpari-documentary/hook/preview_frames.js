// Usage: node preview_frames.js out_dir t1 t2 ...   (saves PNG stills of the hook at given seconds)
const { chromium } = require('playwright'); const path=require('path'), fs=require('fs');
(async()=>{
  const out=process.argv[2]; fs.mkdirSync(out,{recursive:true});
  const b=await chromium.launch({args:['--allow-file-access-from-files']}); const p=await b.newPage({viewport:{width:1920,height:1080}});
  const errs=[]; p.on('pageerror',e=>errs.push(e.message));
  await p.goto('file://'+path.resolve(__dirname,'hook.html')+'?render=1');
  await p.evaluate(()=>Promise.all([window.fontsReady,window.assetsReady]));await p.waitForTimeout(300);
  for(const t of process.argv.slice(3)){
    await p.evaluate(t=>window.drawAt(+t),t);
    const data=await p.evaluate(()=>document.getElementById('c').toDataURL('image/png').split(',')[1]);
    fs.writeFileSync(path.join(out,`f${String(t).padStart(5,'0')}.png`),Buffer.from(data,'base64'));
  }
  console.log('errors:',errs); await b.close();
})();
