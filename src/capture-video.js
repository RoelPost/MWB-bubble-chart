const puppeteer = require('puppeteer');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

const FRAMES_DIR = path.join(__dirname, 'frames');
const OUTPUT_VIDEO = path.join(__dirname, 'nox-bouwplaats-dag1.mp4');
const TOTAL_FRAMES = 144; // 1 dag = 144 intervallen van 10 minuten
const FRAME_DELAY = 150; // ms voor simulatie om te settlen
const VIDEO_FPS = 1; // 1 fps = 1 seconde per frame = realtime 1x snelheid (144 frames = 144 sec)

async function captureVideo() {
    // Maak frames directory
    if (!fs.existsSync(FRAMES_DIR)) {
        fs.mkdirSync(FRAMES_DIR);
    } else {
        // Verwijder oude frames
        fs.readdirSync(FRAMES_DIR).forEach(f => fs.unlinkSync(path.join(FRAMES_DIR, f)));
    }

    console.log('Starting browser...');
    const browser = await puppeteer.launch({
        headless: true,
        args: ['--no-sandbox']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1600, height: 900 });

    console.log('Loading page...');
    await page.goto('http://localhost:8765', { waitUntil: 'networkidle0' });

    // Wacht tot data geladen is en animatie gestart
    await page.waitForSelector('circle');
    await new Promise(r => setTimeout(r, 2000));

    // Pauzeer animatie en reset naar begin
    await page.evaluate(() => {
        pauseAnimation();
        currentTimeIndex = -1; // Wordt 0 bij eerste stap
        resetStateStats();
        resetInsightState();
    });

    console.log(`Capturing ${TOTAL_FRAMES} frames...`);

    for (let i = 0; i < TOTAL_FRAMES; i++) {
        // Stap naar volgend frame
        await page.evaluate(() => {
            currentTimeIndex++;
            updateTimeDisplay();

            // Update nodes
            nodes.forEach(function(node) {
                var machineDataArray = window.machineDataArrays[node.machine_id];
                if (!machineDataArray || machineDataArray.length === 0) return;

                var machineIndex = window.machineDataIndices[node.machine_id];
                machineIndex++;
                if (machineIndex >= machineDataArray.length) machineIndex = 0;
                window.machineDataIndices[node.machine_id] = machineIndex;

                var dataPoint = machineDataArray[machineIndex];
                if (dataPoint) {
                    node.choice = node.machine_type + "_" + dataPoint.power_state;
                    node.power_state = dataPoint.power_state;
                    node.nitrogen_emission = dataPoint.nitrogen_emission;
                    node.nox_gram_per_hour = dataPoint.nox_gram_per_hour;
                    node.nox_gram_per_liter = dataPoint.nox_gram_per_liter;
                    node.verschil_percentage = dataPoint.verschil_percentage;
                }
            });

            updateStatePercentages();
            checkInsightTriggers();
            simulation.alpha(0.5).restart();
        });

        // Wacht tot simulatie settlet
        await new Promise(r => setTimeout(r, FRAME_DELAY));

        // Screenshot
        const frameNum = String(i).padStart(4, '0');
        await page.screenshot({
            path: path.join(FRAMES_DIR, `frame_${frameNum}.png`),
            type: 'png'
        });

        if ((i + 1) % 20 === 0) {
            console.log(`  Frame ${i + 1}/${TOTAL_FRAMES} (${Math.round((i+1)/TOTAL_FRAMES*100)}%)`);
        }
    }

    console.log('Closing browser...');
    await browser.close();

    console.log('Creating video with ffmpeg...');

    // Maak video met ffmpeg (1 fps = realtime 1x snelheid, 144 frames = 2:24 video)
    const ffmpegCmd = `ffmpeg -y -framerate ${VIDEO_FPS} -i "${FRAMES_DIR}/frame_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 18 "${OUTPUT_VIDEO}"`;

    exec(ffmpegCmd, (error, stdout, stderr) => {
        if (error) {
            console.error('ffmpeg error:', error);
            return;
        }
        console.log(`Video saved to: ${OUTPUT_VIDEO}`);

        // Cleanup frames
        fs.readdirSync(FRAMES_DIR).forEach(f => fs.unlinkSync(path.join(FRAMES_DIR, f)));
        fs.rmdirSync(FRAMES_DIR);
        console.log('Frames cleaned up.');
    });
}

captureVideo().catch(console.error);
