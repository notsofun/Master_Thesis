import time, logging, asyncio
from asyncio import Semaphore
from collections import deque
from google.genai import types
from concurrent.futures import ThreadPoolExecutor


# API限流参数
API_LIMITS = {
    'rpm': 150,              # 每分钟请求数
    'daily': 1000,           # 每天请求数
    'rpm_tokens': 2_000_000  # 每分钟输入token数
}

class TokenBucket:
    """令牌桶算法 - 用于精确控制请求率"""
    def __init__(self, capacity, refill_rate):
        """
        capacity: 桶容量
        refill_rate: 每秒补充的令牌数
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
    
    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    async def acquire(self, tokens=1):
        """获取令牌，如果不足则等待"""
        while True:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            
            # 计算需要等待的时间
            wait_time = (tokens - self.tokens) / self.refill_rate
            await asyncio.sleep(min(wait_time, 0.1))  # 最多等待100ms后再检查

class ConcurrencyManager:
    """高级并发管理器 - 基于限流和时间窗口"""
    def __init__(self):
        # RPM (每分钟请求数) 令牌桶
        self.rpm_bucket = TokenBucket(
            capacity=API_LIMITS['rpm'],
            refill_rate=API_LIMITS['rpm'] / 60
        )
        
        # 每分钟输入token数 令牌桶
        self.token_bucket = TokenBucket(
            capacity=API_LIMITS['rpm_tokens'],
            refill_rate=API_LIMITS['rpm_tokens'] / 60
        )
        
        # 请求统计
        self.request_history = deque()  # (timestamp, tokens_used)
        self.error_count = 0
        self.success_count = 0
        self.rate_limit_hits = 0
        
        # 动态并发控制
        self.current_concurrency = 3
        self.min_concurrency = 1
        self.max_concurrency = 8  # 保守估计
        self.semaphore = Semaphore(self.current_concurrency)
        
        # 背压退避参数
        self.backoff_multiplier = 1.0
        self.base_backoff = 1.0
        
    async def acquire(self, estimated_tokens=500):
        """
        获取请求许可
        estimated_tokens: 预估输入token数（用于token bucket检查）
        """
        # 1. 等待并发槽位
        await self.semaphore.acquire()
        
        # 2. 等待RPM令牌
        await self.rpm_bucket.acquire(1)
        
        # 3. 等待输入token令牌
        await self.token_bucket.acquire(estimated_tokens)
        
        logging.debug(f"获得请求许可，当前并发:{self.current_concurrency}，背压倍数:{self.backoff_multiplier:.2f}x")
        
    def release(self):
        """释放并发槽位"""
        self.semaphore.release()
        
    async def on_rate_limit_error(self):
        """触发限流错误时调用"""
        self.rate_limit_hits += 1
        
        # 立即降低并发
        if self.current_concurrency > self.min_concurrency:
            self.current_concurrency = max(self.min_concurrency, self.current_concurrency - 1)
            self.semaphore = Semaphore(self.current_concurrency)
            logging.warning(f"检测到限流错误! 并发数降低到 {self.current_concurrency}，总限流次数: {self.rate_limit_hits}")
        
        # 增加背压倍数
        self.backoff_multiplier = min(3.0, self.backoff_multiplier * 1.5)
        logging.warning(f"背压倍数提升到 {self.backoff_multiplier:.2f}x")
        
        # 等待并逐步恢复
        wait_time = self.base_backoff * self.backoff_multiplier
        logging.warning(f"等待 {wait_time:.2f}秒后重试...")
        await asyncio.sleep(wait_time)
        
    async def on_success(self):
        """成功时调用 - 逐步恢复"""
        self.success_count += 1
        
        # 逐步降低背压倍数 (每20个成功请求)
        if self.success_count % 20 == 0:
            self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.95)
        
        # 逐步增加并发 (每50个成功请求)
        if (self.success_count % 50 == 0 and 
            self.current_concurrency < self.max_concurrency):
            self.current_concurrency += 1
            # 重新创建semaphore
            self.semaphore = Semaphore(self.current_concurrency)
            logging.info(f"并发数增加到 {self.current_concurrency} (背压倍数: {self.backoff_multiplier:.2f}x)")
        
    async def on_other_error(self):
        """其他错误时调用"""
        self.error_count += 1
        
        # 中等程度的背压
        self.backoff_multiplier = min(2.5, self.backoff_multiplier * 1.2)
        wait_time = self.base_backoff * self.backoff_multiplier * 0.5
        logging.warning(f"API错误，背压倍数: {self.backoff_multiplier:.2f}x，等待 {wait_time:.2f}秒...")
        await asyncio.sleep(wait_time)
        
    def get_stats(self):
        """获取统计信息"""
        return {
            'success_count': self.success_count,
            'error_count': self.error_count,
            'rate_limit_hits': self.rate_limit_hits,
            'current_concurrency': self.current_concurrency,
            'backoff_multiplier': self.backoff_multiplier,
            'success_rate': self.success_count / (self.success_count + self.error_count) if (self.success_count + self.error_count) > 0 else 0
        }

class APIRequester:
    """API请求管理器，支持精细化并发控制、令牌管理和限流处理"""
    def __init__(self, client, model='gemini-2.5-pro', max_retries=5):
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.concurrency_manager = ConcurrencyManager()
        self.executor = ThreadPoolExecutor(max_workers=20)
        
    def _estimate_tokens(self, prompt):
        """粗略估算输入token数 (1个字符约0.3个token)"""
        return max(100, int(len(prompt) * 0.3))
        
    async def request_async(self, prompt, temperature=0.8, task_name=""):
        """
        异步API请求，支持限流检测和精细化重试
        """
        estimated_tokens = self._estimate_tokens(prompt)
        await self.concurrency_manager.acquire(estimated_tokens)
        
        try:
            for attempt in range(self.max_retries):
                try:
                    logging.debug(f"[{task_name}] 第{attempt+1}次尝试发送请求 (预估token: {estimated_tokens})")
                    
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        self.executor,
                        lambda: self.client.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config=types.GenerateContentConfig(temperature=temperature)
                        )
                    )
                    
                    await self.concurrency_manager.on_success()
                    logging.debug(f"[{task_name}] API请求成功")
                    return response.text.strip()
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    # 检测限流错误
                    is_rate_limit = any(keyword in error_msg for keyword in [
                        'too many requests',
                        'quota exceeded',
                        'rate limit',
                        '429',
                        '503'
                    ])
                    
                    if is_rate_limit:
                        logging.warning(f"[{task_name}] 触发限流错误: {e}")
                        await self.concurrency_manager.on_rate_limit_error()
                    else:
                        logging.warning(f"[{task_name}] API错误: {e}")
                        await self.concurrency_manager.on_other_error()
                    
                    if attempt < self.max_retries - 1:
                        # 指数退避 + 背压倍数
                        backoff_time = (2 ** attempt) * self.concurrency_manager.backoff_multiplier
                        logging.warning(f"[{task_name}] 等待 {backoff_time:.2f}秒后重试 (第{attempt+1}/{self.max_retries}次)")
                        await asyncio.sleep(backoff_time)
                    else:
                        logging.error(f"[{task_name}] 已达最大重试次数，请求失败")
                        raise
                        
        finally:
            self.concurrency_manager.release()
    
    async def batch_request(self, prompts, task_names=None, temperature=0.8):
        """批量并发请求"""
        if task_names is None:
            task_names = [f"Task-{i}" for i in range(len(prompts))]
            
        tasks = [
            self.request_async(prompt, temperature, name)
            for prompt, name in zip(prompts, task_names)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results