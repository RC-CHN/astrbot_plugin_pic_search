# -*- coding: utf-8 -*-
"""
PicSearch Plugin for AstrBot (Main Entry)

This file contains the main plugin logic, command handling, and coordination
of the scraper, composer, and VLM modules.
"""
import random
from typing import Optional

import astrbot.core.message.components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter, MessageChain
from astrbot.api.message_components import Plain
from astrbot.api.provider import Provider
from astrbot.api.star import Star, Context, register

# Import modularized functions
from .scraper import scrape_image_urls
from .composer import create_collage, _download_image
from .vlm import select_from_collage


@register("astrbot_plugin_pic_search", "RC-CHN", "爬取bing的图片搜索结果，再使用VLM进行选择处理", "v0.1")
class PicSearch(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.vlm_provider_id = self.config.get("vlm_provider_id")
        self.batch_size = self.config.get("batch_size", 16)
        self.default_scrape_count = self.config.get("default_scrape_count", 64)

    def _get_vlm_provider(self) -> Optional[Provider]:
        """Gets the provider instance for VLM operations."""
        if self.vlm_provider_id:
            provider = self.context.get_provider_by_id(self.vlm_provider_id)
            if not provider:
                logger.error(f"PicSearch: Could not find a provider with ID '{self.vlm_provider_id}'.")
                return None
            return provider
        return self.context.llm

    @filter.command("搜图", ["picsearch"])
    async def handle_pic_search(self, event: AstrMessageEvent, query: str, description: str, count: Optional[int] = None):
        logger.info("--- EXECUTING COMMAND HANDLER: handle_pic_search ---")
        async for result in self._do_pic_search(event, query, description, count):
            yield result
    @filter.llm_tool(name="pic_search")
    async def pic_search_tool(self, event: AstrMessageEvent, query: str, description: str, count: int = 64):
        '''根据关键词和描述在网络上搜索一张最匹配的图片。

        本工具首先会根据通用关键词（query）抓取大量图片，然后利用视觉语言模型（VLM）根据你的详细描述（description）进行智能筛选，最终返回最符合要求的一张。
        
        Args:
            query(string): **必需参数**。用于初步搜索图片的通用关键词，例如“猫”或“东京街景”。
            description(string): **必需参数**。对期望图片的具体、详细的视觉描述。**如果用户输入模糊，你必须根据 query 和上下文自行生成一个合理的、具体的视觉描述**，例如“一只戴着帽子的黑猫”或“雨夜的涩谷街头，霓虹灯闪烁”。
            count(number): 初始抓取的图片数量，用于扩大筛选范围。这是一个可选参数，默认是64，不建议更小，但过大会增加带宽压力，建议为16的倍数。
        '''
        logger.info("--- EXECUTING LLM TOOL: pic_search_tool ---")
        # LLM Tool implementation
        total_count = count

        vlm_provider = self._get_vlm_provider()
        if not vlm_provider:
            return "无法获取有效的VLM Provider，请检查插件配置或当前会话的LLM设置。"

        # Notify user that the process has started
        # await event.send(MessageChain([Plain(f"收到任务！\n- 搜索: {query}\n- 要求: {description}\n- 数量: {total_count}\n正在后台处理，请稍候...")]))

        try:
            # 1. Scrape image URLs
            image_urls = await scrape_image_urls(query, total_count)
            if not image_urls:
                return "未能抓取到任何图片链接，任务终止。"

            logger.info(f"PicSearch: 成功抓取 {len(image_urls)} 个链接。开始进行淘汰赛筛选...")

            # 2. Run the tournament process
            winner_url = await self._process_in_batches(image_urls, description, vlm_provider)
            if not winner_url:
                return "筛选过程没有产生最终结果，任务终止。"

            # 3. Download and send the final image
            final_image_bytes = await _download_image(winner_url)
            if final_image_bytes:
                # Send image directly
                result_image = Comp.Image.fromBytes(final_image_bytes)
                await event.send(MessageChain([result_image]))
                # Return text summary to LLM
                return "已成功为用户找到并发送了图片。"
            else:
                return f"无法下载最终图片，链接为：{winner_url}"

        except Exception as e:
            logger.error(f"PicSearch: An unexpected error occurred in tool handler: {e}", exc_info=True)
            return "处理过程中发生严重错误，详情请查看日志。"


    async def _do_pic_search(self, event: AstrMessageEvent, query: str, description: str, count: Optional[int]):
        total_count = count if count is not None else self.default_scrape_count

        vlm_provider = self._get_vlm_provider()
        if not vlm_provider:
            yield event.plain_result("无法获取有效的VLM Provider，请检查插件配置或当前会话的LLM设置。")
            return

        yield event.plain_result(f"收到任务！\n- 搜索: {query}\n- 要求: {description}\n- 数量: {total_count}\n正在后台处理，请稍候...")

        try:
            # 1. Scrape image URLs
            image_urls = await scrape_image_urls(query, total_count)
            if not image_urls:
                yield event.plain_result("未能抓取到任何图片链接，任务终止。")
                return

            logger.info(f"PicSearch: 成功抓取 {len(image_urls)} 个链接。开始进行淘汰赛筛选...")

            # 2. Run the tournament process
            winner_url = await self._process_in_batches(image_urls, description, vlm_provider)
            if not winner_url:
                yield event.plain_result("筛选过程没有产生最终结果，任务终止。")
                return

            # 3. Send the final result
            final_image_bytes = await _download_image(winner_url)
            if final_image_bytes:
                yield event.chain_result([
                    Comp.Plain("为您找到了最匹配的图片："),
                    Comp.Image.fromBytes(final_image_bytes)
                ])
            else:
                yield event.plain_result(f"无法下载最终图片，链接为：{winner_url}")

        except Exception as e:
            logger.error(f"PicSearch: An unexpected error occurred in main handler: {e}", exc_info=True)
            yield event.plain_result("处理过程中发生严重错误，详情请查看日志。")

    async def _process_in_batches(self, image_urls: list, description: str, vlm_provider: Provider) -> Optional[str]:
        """The core tournament-style selection process. Logs progress instead of replying."""
        current_winners = image_urls
        round_num = 1
        stalemate_counter = 0
        while len(current_winners) > 1:
            logger.info(f"PicSearch: --- 第 {round_num} 轮淘汰赛开始，当前选手: {len(current_winners)} 名 ---")
            next_round_winners = []
            
            # Determine the prompt for this round
            effective_prompt = description
            prompt_enhancements = []

            # Final round logic: if 16 or fewer candidates, activate stricter prompt
            if len(current_winners) <= 16:
                logger.info(f"PicSearch: Entering final round with {len(current_winners)} candidates. Activating stricter prompt.")
                prompt_enhancements.append(
                    "重要指示：你正处于决赛圈。请对以下图片进行严格比较，并只选出其中最符合描述的一半（向上取整）作为优胜者。如果所有图片都同样优秀，也请务必只选择一半。"
                )

            # Stalemate logic
            if stalemate_counter > 0:
                logger.warning(f"PicSearch: Activating enhanced prompt due to stalemate (counter: {stalemate_counter}).")
                prompt_enhancements.append(
                    "重要指示: 你必须进行筛选。请从以上图片中，严格挑选出一张或几张最符合描述的图片。"
                    "如果所有图片都符合，请只选择最优秀的一张。"
                )
            
            if prompt_enhancements:
                effective_prompt = f"{description}\n\n{' '.join(prompt_enhancements)}"

            for i in range(0, len(current_winners), self.batch_size):
                batch_urls = current_winners[i:i + self.batch_size]
                if not batch_urls: continue

                logger.info(f"PicSearch: 正在处理本轮第 {i // self.batch_size + 1} 批，共 {len(batch_urls)} 张图片...")
                
                collage_bytes, successful_urls = await create_collage(batch_urls)
                if not collage_bytes or not successful_urls:
                    logger.warning("PicSearch: 本批次图片下载或合成失败，跳过。")
                    continue

                logger.info("PicSearch: 拼接图已生成，正在提交给VLM进行筛选...")

                selected_indices = await select_from_collage(collage_bytes, effective_prompt, vlm_provider)
                if not selected_indices:
                    logger.warning("PicSearch: VLM未能从本批次中选出任何图片。")
                    continue
                
                batch_winners = []
                for index in selected_indices:
                    actual_index = index - 1
                    if 0 <= actual_index < len(successful_urls):
                        batch_winners.append(successful_urls[actual_index])
                
                if batch_winners:
                    logger.info(f"PicSearch: 本批次优胜者: {len(batch_winners)} 名。")
                    next_round_winners.extend(batch_winners)

            if not next_round_winners:
                logger.error("PicSearch: Tournament round produced no winners.")
                return None

            previous_winner_count = len(current_winners)
            current_winners = list(set(next_round_winners))
            
            if len(current_winners) == previous_winner_count and len(current_winners) > 1:
                stalemate_counter += 1
                logger.warning(f"PicSearch: Stalemate detected (Round {round_num}). Count: {len(current_winners)}. Counter: {stalemate_counter}")
            else:
                stalemate_counter = 0

            if stalemate_counter >= 2:
                logger.error("PicSearch: Stalemate persists after prompt modification. Forcing a random choice.")
                current_winners = [random.choice(current_winners)]
                break
            
            round_num += 1

        return current_winners[0] if current_winners else None
