# src/agents/ocr_agent.py
"""
OCR Agent for multi_agents_demo project

Integrates DBNet + CRNN for end-to-end OCR
"""

import asyncio
import logging
from typing import List, Dict
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OCRAgent:
    """OCR Agent - 完整的文本检测与识别"""
    
    def __init__(
        self,
        det_model_path: str,
        rec_model_path:  str,
        device: str = 'cuda'
    ):
        from models.ocr_system import OCRSystem
        
        self.ocr_system = OCRSystem(
            det_model_path=det_model_path,
            rec_model_path=rec_model_path,
            rec_type='crnn',
            device=device
        )
        logger.info("OCR Agent initialized")
    
    async def execute(self, task:  Dict) -> Dict:
        """执行 OCR 任务
        
        Args:
            task: {
                'image_path': str,
                'visualize': bool,
                'output_path': str (optional)
            }
        
        Returns:
            {
                'results': List[Dict],
                'count': int,
                'status': str
            }
        """
        image_path = task['image_path']
        
        # OCR 识别（在线程池中运行，避免阻塞）
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self. ocr_system,
            image_path
        )
        
        # 可视化
        if task. get('visualize', False):
            output_path = task. get('output_path', 'ocr_result.jpg')
            self.ocr_system.visualize(image_path, results, output_path)
        
        logger.info(f"OCR completed:  {len(results)} text regions")
        
        return {
            'results': results,
            'count': len(results),
            'status': 'success'
        }
    
    def get_capabilities(self) -> List[str]:
        """返回Agent能力"""
        return [
            'text_detection',
            'text_recognition',
            'ocr',
            'document_analysis'
        ]
