# import asyncio
# from playwright.async_api import async_playwright
# from langchain.tools import BaseTool

# class PlaywrightBrowserTool(BaseTool):
#     """
#     A LangChain Tool that uses Playwright to navigate to a given URL and
#     return the page's HTML content.
#     """
#     name: str = "browse_url"
#     description: str = (
#         "Navigate to a URL using Playwright and return the full HTML content of the page."
#     )
#     print(description)

#     async def _run(self, url: str) -> str:
#         async with async_playwright() as pw:
#             browser = await pw.chromium.launch(headless=True)
#             page = await browser.new_page()
#             await page.goto(url)
#             content = await page.content()
#             await browser.close()
#             return content

#     async def _arun(self, url: str) -> str:
#         # For LangChain async compatibility
#         return await self._run(url)

# # Wrap the tool for synchronous use in an agent if needed
# from langchain.agents import Tool

# def sync_browse(url: str) -> str:
#     return asyncio.get_event_loop().run_until_complete(
#         PlaywrightBrowserTool()._run(url)
#     )

# browser_tool = Tool(
#     name="browse_url",
#     func=sync_browse,
#     description="Navigate to a URL and return its HTML content."
# )

# src/browser_tool.py
# src/browser_tool.py

# from langchain_community.agent_toolkits.playwright import PlayWrightBrowserToolkit
# from langchain_community.tools.playwright.utils import create_sync_playwright_browser

# # spin up a sync Playwright browser
# sync_browser = create_sync_playwright_browser()

# # instantiate the toolkit (only sync/async browser args are allowed)
# toolkit = PlayWrightBrowserToolkit.from_browser(
#     sync_browser=sync_browser
# )

# # export the list of tools for initialize_agent()
# browser_tools = toolkit.get_tools()
