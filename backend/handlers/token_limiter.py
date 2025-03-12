import os
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading

# Get logger for this module
logger = logging.getLogger('token_limiter')

class TokenLimiter:
    """
    A class to track and limit token usage for Azure API calls.
    Implements a daily token limit to prevent excessive API usage.
    """
    def __init__(self, daily_limit=100000, storage_path=None):
        """
        Initialize the token limiter with a daily token limit.
        
        Args:
            daily_limit (int): Maximum number of tokens allowed per day
            storage_path (str, optional): Path to store token usage data
        """
        self.daily_limit = daily_limit
        self.lock = threading.Lock()
        
        logger.info(f"Initializing TokenLimiter with daily limit: {daily_limit}")
        
        # Set default storage path if not provided
        if storage_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.storage_path = base_dir / "assets" / "token_usage.json"
        else:
            self.storage_path = Path(storage_path)
            
        logger.info(f"Token usage data will be stored at: {self.storage_path}")
        
        # Ensure the directory exists
        self.storage_path.parent.mkdir(exist_ok=True)
        
        # Initialize usage data if it doesn't exist
        if not self.storage_path.exists():
            logger.info("Token usage data file does not exist, initializing...")
            self._initialize_usage_data()
        else:
            logger.info("Token usage data file exists, loading existing data")
            usage_data = self._load_usage_data()
            logger.info(f"Current usage data: {usage_data}")
    
    def _initialize_usage_data(self):
        """Initialize the token usage data file with default values."""
        today = datetime.now().strftime("%Y-%m-%d")
        usage_data = {
            "last_reset_date": today,
            "daily_usage": 0,
            "total_usage": 0
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(usage_data, f)
        
        logger.info(f"Initialized token usage data: {usage_data}")
    
    def _load_usage_data(self):
        """Load token usage data from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                logger.debug(f"Loaded token usage data: {data}")
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading token usage data: {e}")
            logger.info("Reinitializing token usage data")
            self._initialize_usage_data()
            with open(self.storage_path, 'r') as f:
                return json.load(f)
    
    def _save_usage_data(self, usage_data):
        """Save token usage data to storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(usage_data, f)
        logger.debug(f"Saved token usage data: {usage_data}")
    
    def _reset_daily_usage_if_needed(self, usage_data):
        """Reset daily usage if it's a new day."""
        today = datetime.now().strftime("%Y-%m-%d")
        last_reset = usage_data.get("last_reset_date", "")
        
        if today != last_reset:
            logger.info(f"Resetting daily usage. Last reset: {last_reset}, Today: {today}")
            usage_data["last_reset_date"] = today
            usage_data["daily_usage"] = 0
            return True
        return False
    
    def check_limit(self):
        """
        Check if the token limit has been reached.
        
        Returns:
            bool: True if limit not reached, False if limit reached
        """
        with self.lock:
            usage_data = self._load_usage_data()
            self._reset_daily_usage_if_needed(usage_data)
            
            remaining = self.daily_limit - usage_data["daily_usage"]
            result = usage_data["daily_usage"] < self.daily_limit
            
            logger.info(f"Checking token limit: daily_usage={usage_data['daily_usage']}, limit={self.daily_limit}, remaining={remaining}, limit_reached={not result}")
            
            return result
    
    def add_tokens(self, token_count):
        """
        Add tokens to the usage counter.
        
        Args:
            token_count (int): Number of tokens to add
            
        Returns:
            bool: True if tokens were added successfully, False if limit exceeded
        """
        with self.lock:
            usage_data = self._load_usage_data()
            reset = self._reset_daily_usage_if_needed(usage_data)
            
            # Update token counts
            old_daily = usage_data["daily_usage"]
            old_total = usage_data["total_usage"]
            
            usage_data["daily_usage"] += token_count
            usage_data["total_usage"] += token_count
            
            logger.info(f"Added {token_count} tokens. Daily usage: {old_daily} -> {usage_data['daily_usage']}, Total usage: {old_total} -> {usage_data['total_usage']}")
            
            # Save updated data
            self._save_usage_data(usage_data)
            return True
    
    def get_usage_stats(self):
        """
        Get current token usage statistics.
        
        Returns:
            dict: Token usage statistics
        """
        with self.lock:
            usage_data = self._load_usage_data()
            self._reset_daily_usage_if_needed(usage_data)
            
            stats = {
                "daily_usage": usage_data["daily_usage"],
                "daily_limit": self.daily_limit,
                "total_usage": usage_data["total_usage"],
                "remaining_tokens": max(0, self.daily_limit - usage_data["daily_usage"]),
                "last_reset_date": usage_data["last_reset_date"]
            }
            
            logger.info(f"Token usage stats: {stats}")
            
            return stats

# Create a singleton instance
token_limiter = TokenLimiter(
    daily_limit=int(os.getenv("AZURE_DAILY_TOKEN_LIMIT", "100000"))
)

# Log the initial token limit
logger.info(f"TokenLimiter initialized with daily limit: {token_limiter.daily_limit}") 