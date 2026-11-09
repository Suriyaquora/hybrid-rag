// Renders hook.html deterministically to MP4 (1080p30) with the synthesised score.
// Usage: node render.js out.mp4 music.wav [fps=30] [startSec=0] [endSec=180]
const { chromium } = require('playwright'); const { spawn } = require('child_process'); const path = require('path');
(async () => {
  const [out, wav, fpsArg, s0, s1] = process.argv.slice(2); const fps = +(fpsArg || 30), a = +(s0 || 0), b = +(s1 || 180);
  const br = await chromium.launch({args:['--allow-file-access-from-files']}); const pg = await br.newPage({ viewport: { width: 1920, height: 1080 } });
  await pg.goto('file://' + path.resolve(__dirname, 'hook.html') + '?render=1'); await pg.evaluate(() => Promise.all([window.fontsReady,window.assetsReady])); await pg.waitForTimeout(300);
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
    '-ss', String(a), '-t', String(b - a), '-i', wav, '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  const total = Math.round((b - a) * fps); const t0 = Date.now();
  for (let i = 0; i < total; i++) {
    const t = a + i / fps;
    const b64 = await pg.evaluate(t => { window.drawAt(t); return document.getElementById('c').toDataURL('image/jpeg', 0.93).split(',')[1]; }, t);
    if (!ff.stdin.write(Buffer.from(b64, 'base64'))) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log(`frame ${i}/${total}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end(); await new Promise(r => ff.on('close', r)); await br.close(); console.log('done', out);
})();
