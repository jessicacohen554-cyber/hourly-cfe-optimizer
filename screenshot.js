const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const path = require('path');

(async () => {
    const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'] });
    const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });

    const filePath = 'file://' + path.resolve(__dirname, 'banner-previews.html');
    await page.goto(filePath, { waitUntil: 'networkidle' });

    // Wait for fonts to load
    await page.waitForTimeout(3000);

    // Screenshot each banner preview card
    const bannerCards = await page.$$('.preview-card');
    for (let i = 0; i < bannerCards.length; i++) {
        await bannerCards[i].screenshot({
            path: path.resolve(__dirname, `previews/banner-${i + 1}.png`)
        });
        console.log(`Saved banner-${i + 1}.png`);
    }

    // Screenshot each typography card
    const typoCards = await page.$$('.typo-card');
    for (let i = 0; i < typoCards.length; i++) {
        await typoCards[i].screenshot({
            path: path.resolve(__dirname, `previews/font-${i + 1}.png`)
        });
        console.log(`Saved font-${i + 1}.png`);
    }

    // Full page screenshot
    await page.screenshot({
        path: path.resolve(__dirname, `previews/all-previews.png`),
        fullPage: true
    });
    console.log('Saved all-previews.png');

    await browser.close();
})();
