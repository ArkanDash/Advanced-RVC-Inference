import asyncio
import glob
import os
import subprocess
from pathlib import Path

import aiohttp
from playwright.async_api import Page, async_playwright

from metrics.metrics import metrics_counter_inc
from neo.utils import logger

_BEDROCK_PROJECT = os.environ.get("BEDROCK_PROJECT", "")


def is_bedrock_env() -> bool:
    return _BEDROCK_PROJECT != ""


def find_chromium_executable() -> str:
    """
    自动检测 Chromium 可执行文件路径。
    优先级：
    1. 环境变量 CHROMIUM_PATH
    2. Playwright 安装的 Chromium
    3. 系统安装的 Chromium/Chrome
    """
    # 1. 检查环境变量
    chromium_path = os.environ.get("CHROMIUM_PATH")
    if chromium_path and os.path.isfile(chromium_path):
        logger.info(f"[GlobalBrowser] 使用环境变量指定的 Chromium: {chromium_path}")
        return chromium_path

    # 2. 搜索 Playwright 安装的 Chromium
    # Playwright 默认安装路径
    playwright_paths = [
        os.path.expanduser("~/.cache/ms-playwright"),  # Linux 用户目录
        "/root/.cache/ms-playwright",  # Linux root 用户
        "/home/minimax/.cache/ms-playwright",  # minimax 用户
        os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""),  # 自定义路径
    ]

    for base_path in playwright_paths:
        if not base_path or not os.path.isdir(base_path):
            continue
        # Playwright Chromium 路径模式: chromium-*/chrome-linux/chrome
        pattern = os.path.join(base_path, "chromium-*", "chrome-linux", "chrome")
        matches = glob.glob(pattern)
        if matches:
            # 选择最新版本（按字母排序取最后一个）
            chromium_path = sorted(matches)[-1]
            if os.path.isfile(chromium_path):
                logger.info(f"[GlobalBrowser] 找到 Playwright 安装的 Chromium: {chromium_path}")
                return chromium_path

    # 3. 系统安装的浏览器
    system_browsers = [
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/opt/google/chrome/chrome",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
    ]

    for browser_path in system_browsers:
        if os.path.isfile(browser_path):
            logger.info(f"[GlobalBrowser] 找到系统浏览器: {browser_path}")
            return browser_path

    # 找不到浏览器
    raise FileNotFoundError(
        "未找到 Chromium 浏览器。请确保已安装 Playwright Chromium (npx playwright install chromium) "
        "或设置 CHROMIUM_PATH 环境变量指向 Chromium 可执行文件。"
    )


async def handle_new_page(page: Page):
    """
    Handle new page events and execute custom logic
    """
    print(f"New page created: {page.url}")


async def launch_chrome_debug(use_chrome_channel: bool = False, headless: bool = False):
    """
    Launch Chrome browser with remote debugging enabled on port 9222
    Returns the browser instance when launched successfully
    """
    try:
        extension_path = Path(os.path.dirname(__file__)).joinpath("browser_extension/error_capture")  # type: ignore
        playwright = await async_playwright().start()

        workspace = "/workspace" if is_bedrock_env() else "./workspace"
        user_data_dir = os.path.join(workspace, "browser", "user_data")

        # 删除浏览器单例锁文件（如果存在），避免从NAS恢复的旧锁文件导致冲突
        # 使用 lexists 而不是 exists，因为这些文件可能是指向不存在目标的符号链接
        singleton_files = ["SingletonLock", "SingletonSocket", "SingletonCookie"]
        for filename in singleton_files:
            file_path = os.path.join(user_data_dir, filename)
            try:
                if os.path.lexists(file_path):
                    os.remove(file_path)
                    logger.info(f"已删除浏览器单例文件: {file_path}")
            except Exception as e:
                logger.warning(f"删除浏览器单例文件失败 {file_path}: {str(e)}")

        # 检查是否已有 Chrome 实例在 9222 端口运行
        logger.info("[GlobalBrowser] Checking if Chrome is already running on port 9222...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:9222/json/version", timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        logger.info("[GlobalBrowser] Chrome is already running on port 9222, reusing existing instance")
                        browser = await playwright.chromium.connect_over_cdp("http://localhost:9222")
                        context = browser.contexts[0] if browser.contexts else await browser.new_context()
                        metrics_counter_inc("agent_browser_launch", {"status": "success"})

                        # 监听新页面事件
                        context.on("page", handle_new_page)
                        for page in context.pages:
                            await handle_new_page(page)

                        # Keep browser process alive
                        while True:
                            await asyncio.sleep(1000)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.info("[GlobalBrowser] No existing Chrome instance found, starting a new one...")

        # 准备 Chrome 启动参数
        chrome_args = [
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--disable-background-timer-throttling",
            "--disable-popup-blocking",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-window-activation",
            "--disable-focus-on-load",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-position=0,0",
            "--disable-web-security",
            "--disable-site-isolation-trials",
            "--disable-features=IsolateOrigins,site-per-process",
            f"--disable-extensions-except={extension_path}",
            f"--load-extension={extension_path}",
            "--remote-debugging-port=9222",
            "--remote-debugging-address=127.0.0.1",  # 仅允许本地访问，防止外部连接
        ]

        # 使用 subprocess.Popen 启动 Chrome
        chromium_path = find_chromium_executable()
        logger.info(f"[GlobalBrowser] Starting Chrome ({chromium_path}) with remote debugging on port 9222...")
        subprocess.Popen(
            [chromium_path] + chrome_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=workspace,
        )

        # 等待 Chrome 启动并暴露 CDP 端口
        logger.info("[GlobalBrowser] Waiting for Chrome to be ready...")
        max_wait_time = 30
        poll_interval = 1
        waited = 0
        chrome_ready = False

        while waited < max_wait_time:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://localhost:9222/json/version", timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            logger.info(f"[GlobalBrowser] Chrome is ready after {waited} seconds ✓")
                            chrome_ready = True
                            break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass

            waited += poll_interval
            logger.debug(f"[GlobalBrowser] Still waiting for Chrome... ({waited}/{max_wait_time}s)")

        if not chrome_ready:
            logger.warning(f"[GlobalBrowser] Chrome may not be ready after {max_wait_time} seconds, proceeding anyway...")

        # 连接到 Chrome
        logger.info("[GlobalBrowser] Connecting to Chrome via CDP...")
        browser = await playwright.chromium.connect_over_cdp(
            "http://localhost:9222",
            timeout=30000,  # 30 second timeout for connection
        )
        logger.info("[GlobalBrowser] Successfully connected to Chrome ✓")

        # 创建或获取 browser context
        if browser.contexts:
            context = browser.contexts[0]
        else:
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_data_dir=user_data_dir,
            )

        metrics_counter_inc("agent_browser_launch", {"status": "success"})

        # 监听新页面事件
        context.on("page", handle_new_page)

        # 处理已经打开的页面
        for page in context.pages:
            await handle_new_page(page)

        # Keep browser process alive
        while True:
            await asyncio.sleep(1000)

    except Exception as e:
        logger.exception(f"Failed to launch Chrome browser: {str(e)}")
        metrics_counter_inc("agent_browser_launch", {"status": "failed"})
        raise


if __name__ == "__main__":
    asyncio.run(launch_chrome_debug())
