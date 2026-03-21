import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    img_dir = '/Users/studio/Downloads/project/Smart_vision/docs/images'
    os.makedirs(img_dir, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1280, 'height': 800})
        page = await context.new_page()
        
        try:
            print("Navigating to home...")
            await page.goto('http://localhost:5173/')
            await page.wait_for_load_state('networkidle')
            
            if 'login' in page.url:
                print("Logging in...")
                try:
                    await page.get_by_placeholder("admin").fill("admin", timeout=5000)
                    await page.get_by_placeholder("••••••••").fill("admin123")
                except:
                    await page.locator('input[type="text"]').fill("admin")
                    await page.locator('input[type="password"]').fill("admin123")
                await page.locator('button[type="submit"]').click()
                await page.wait_for_load_state('networkidle')
                await page.wait_for_timeout(2000)
            
            print("Taking index UI screenshot...")
            await page.goto('http://localhost:5173/app/index')
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(2000)
            await page.screenshot(path=os.path.join(img_dir, 'fig_index_ui.png'))
            
            print("Taking catalog UI screenshot...")
            await page.goto('http://localhost:5173/app/catalog')
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(2000)
            await page.screenshot(path=os.path.join(img_dir, 'fig_catalog_ui.png'))

            print("Screenshots captured successfully.")
        except Exception as e:
            print(f"Error during capture: {e}")
        finally:
            await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
