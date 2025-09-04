"""
Semantic Scholar API Client
Standalone client for interacting with Semantic Scholar API with rate limiting.
"""

import requests
from requests import Response
import time
from threading import Lock
from collections import deque
from typing import Optional, List, Dict
import pandas as pd


class SemanticScholarClient:
    """Stateful client for Semantic Scholar API with rate limiting"""
    
    def __init__(self, api_key: Optional[str] = None, max_requests_per_second: int = 1, 
                 max_requests_per_day: int = 10_000, base_url: str = "https://api.semanticscholar.org/graph/v1"):
        self.api_key = api_key
        self.max_requests_per_second = max_requests_per_second
        self.max_requests_per_day = max_requests_per_day
        self.base_url = base_url
        
        # Thread-safe rate limiting
        self._lock = Lock()
        self._request_times = deque()
        self._daily_request_count = 0
        self._daily_reset_time = time.time() + 86400  # 24 hours from now
        
        self.headers = {}
        self.headers["x-api-key"] = self.api_key
        self.max_requests_per_second == 1
        self.max_requests_per_day = 10_000

    def _check_rate_limits(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        
        with self._lock:
            # Reset daily counter if 24 hours have passed
            if current_time > self._daily_reset_time:
                self._daily_request_count = 0
                self._daily_reset_time = current_time + 86400
            
            # Check daily limit
            if self._daily_request_count >= self.max_requests_per_day:
                raise Exception(f"Daily rate limit of {self.max_requests_per_day} requests exceeded")
            
            # Remove requests older than 1 second
            while self._request_times and current_time - self._request_times[0] > 1.0:
                self._request_times.popleft()
            
            # Check per-second limit
            if len(self._request_times) >= self.max_requests_per_second:
                sleep_time = 1.0 - (current_time - self._request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Remove the old request after sleeping
                    self._request_times.popleft()
            
            # Record this request
            self._request_times.append(current_time)
            self._daily_request_count += 1

    def request(self, endpoint: str, params: Optional[dict] = None, method: str = "GET", json_data: Optional[dict] = None) -> Response:
        """Make a rate-limited request to Semantic Scholar API"""
        self._check_rate_limits()
        
        # Prepare parameters
        if params is None:
            params = {}
        
        # Make the request
        url = f"{self.base_url}/{endpoint}"
        
        response = requests.get(url, params=params, headers=self.headers)
        
        # Handle rate limit responses
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))
            print(f"Rate limited by Semantic Scholar. Waiting {retry_after} seconds before retry.")
            time.sleep(retry_after)
            return self.request(endpoint, params, method, json_data)  # Retry once
        
        if response.status_code == 404:
            print(f"Resource not found: {endpoint}")
            return response  # Return 404 response for handling by caller
        
        response.raise_for_status()
        return response
    
    def get_snippet(
        self, 
        query: str = None, 
        venue: str = None, 
        year: str = None, 
        minCitationCount: int = 5,
        fields: List[str] = None) -> Optional[dict]:
        """Get text snippets by venue and year"""
        
        if fields is None:
            fields = ["snippet.text","snippet.section","snippet.annotations.sentences","snippet.snippetKind"]
        
        params = {
            "query": query,
            "fields": ",".join(fields),
            "minCitationCount": minCitationCount,
            "limit": 1_000, # max
            "year": year,
            "venue": venue
            }
        response = self.request(f"snippet/search", params)
        
        if response.status_code == 404:
            return None
        
        return response.json()
    

