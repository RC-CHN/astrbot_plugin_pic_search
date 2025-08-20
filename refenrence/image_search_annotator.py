# -*- coding: utf-8 -*-
"""
一个使用 Playwright 的 Python 脚本，用于在图片搜索引擎上执行搜索，
并用数字标签标注搜索结果中的图片。
"""

import argparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

def annotate_image_search(query: str, headless: bool = True):
    """
    启动浏览器，搜索图片，用数字标注它们，并截取屏幕截图。

    :param query: 要搜索的图片关键词。
    :param headless: 是否以无头模式运行浏览器。
    """
    with sync_playwright() as p:
        # 启动浏览器实例
        # 在调试时可以设置 headless=False 以便观察浏览器操作
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            # 1. 导航到必应图片搜索
            print("正在导航到必应图片...")
            page.goto("https://www.bing.com/images/search", timeout=60000)

            # 2. 找到搜索框，输入关键词并提交
            print(f"正在搜索关键词: '{query}'...")
            # 使用 #sb_form_q 作为搜索框的选择器
            search_box_selector = "#sb_form_q"
            page.wait_for_selector(search_box_selector, state="visible", timeout=10000)
            page.fill(search_box_selector, query)
            page.press(search_box_selector, "Enter")

            # 3. 等待搜索结果加载
            print("等待搜索结果加载...")
            # 等待第一个图片容器出现，这是一个好迹象，表明结果正在加载
            # 选择器 'div.img_cont' 匹配包含图片的容器
            image_container_selector = "div.img_cont"
            page.wait_for_selector(image_container_selector, state="visible", timeout=15000)
            # 等待一小段时间，让更多的图片加载进来
            page.wait_for_timeout(3000)

            # 4. 核心功能: 注入 JavaScript 来标注图片
            print("正在执行 JavaScript 以标注图片...")
            
            # 这段 JavaScript 代码会找到页面上所有的图片，并在每张图片的左上角添加一个数字标签。
            # 这些标签被设计成高对比度和清晰可见，以便后续的 VLM 模型处理。
            js_code = """
            () => {
                // 选择所有代表搜索结果的图片元素 (img.mimg 是一个比较可靠的选择器)
                const images = document.querySelectorAll('div.img_cont img.mimg');
                
                // 遍历每张图片并添加一个带样式的数字标签
                images.forEach((img, index) => {
                    // 创建标签元素
                    const label = document.createElement('div');
                    
                    // 获取图片在视口中的位置和尺寸
                    const rect = img.getBoundingClientRect();

                    // 设置标签文本
                    label.innerText = `${index + 1}`;
                    
                    // --- 标签样式 ---
                    // 设计目标：高可见性，以便 VLM 识别
                    label.style.position = 'absolute';
                    // 将标签定位到图片的左上角（考虑页面滚动）
                    label.style.top = `${window.scrollY + rect.top}px`;
                    label.style.left = `${window.scrollX + rect.left}px`;
                    label.style.backgroundColor = 'rgba(255, 20, 20, 0.85)'; // 鲜艳的红色背景
                    label.style.color = 'white'; // 白色文字
                    label.style.padding = '3px 6px'; // 内边距
                    label.style.fontSize = '16px'; // 清晰的字体大小
                    label.style.fontWeight = 'bold'; // 粗体
                    label.style.border = '2px solid white'; // 白色边框增加对比度
                    label.style.borderRadius = '4px'; // 圆角
                    label.style.zIndex = '9999'; // 确保标签在最上层
                    label.style.fontFamily = 'Arial, sans-serif'; // 通用字体
                    label.style.boxShadow = '0 0 5px rgba(0,0,0,0.5)'; // 添加阴影以突出
                    
                    // 将标签添加到文档中
                    document.body.appendChild(label);
                });
                
                // 返回找到的图片数量
                return images.length;
            }
            """
            
            # 执行 JavaScript 并获取标注的图片数量
            annotated_count = page.evaluate(js_code)
            print(f"成功标注了 {annotated_count} 张图片。")

            # 5. 截取整个页面的截图
            screenshot_path = "labeled_screenshot.png"
            print(f"正在截取全屏并保存到 '{screenshot_path}'...")
            page.screenshot(path=screenshot_path, full_page=True)
            print("截图成功保存。")
            
            page.wait_for_timeout(3000)

        except PlaywrightTimeoutError as e:
            print(f"操作超时: {e}")
            print("请检查您的网络连接，或尝试增加超时时间。")
            page.screenshot(path="error_screenshot.png")
            print("已保存当前页面的截图 'error_screenshot.png' 以供调试。")
        except Exception as e:
            print(f"发生未知错误: {e}")
            page.screenshot(path="error_screenshot.png")
            print("已保存当前页面的截图 'error_screenshot.png' 以供调试。")
        finally:
            # 6. 关闭浏览器
            browser.close()
            print("浏览器已关闭。")

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="使用 Playwright 在必应图片上搜索并用数字标签标注结果。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "query", 
        type=str, 
        help="要搜索的图片关键词。\n例如: 'cute cats'"
    )
    parser.add_argument(
        "--visible",
        action="store_false",
        dest="headless",
        help="运行浏览器时显示 GUI 界面，用于调试。"
    )
    
    args = parser.parse_args()
    
    # 运行主函数
    annotate_image_search(args.query, headless=args.headless)